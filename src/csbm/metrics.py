import os
from typing import Any, List, Literal, Optional, Union

import torch
from torch import nn
from torch.nn import functional as F

from torchmetrics import Metric
from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.image.fid import FrechetInceptionDistance as FID
from torchmetrics.text import EditDistance, Perplexity, BLEUScore
from torchmetrics.functional.text.perplexity import _perplexity_update

from transformers import (
    CLIPImageProcessor, 
    CLIPVisionModelWithProjection, 
    AutoModelForCausalLM, 
    AutoTokenizer,
    pipeline
)

FID_WEIGHTS_URL = 'https://github.com/mseitzer/pytorch-fid/releases/download/fid_weights/pt_inception-2015-12-05-6726825d.pth'  # noqa: E501
CLIP_MODEL_NAME = "openai/clip-vit-large-patch14-336"


def _resize_bicubic(images, size):
    """Resize images using bicubic interpolation."""
    images = torch.from_numpy(images.transpose(0, 3, 1, 2))
    images = torch.nn.functional.interpolate(images, size=(size, size), mode="bicubic")
    images = images.permute(0, 2, 3, 1).numpy()
    return images


class CLIPFeatureExtractor(nn.Module):
    """Custom CLIP image embedding calculator adapted from https://github.com/sayakpaul/cmmd-pytorch."""

    def __init__(self, model_name=CLIP_MODEL_NAME):
        super().__init__()
        self.image_processor = CLIPImageProcessor.from_pretrained(model_name)
        self.model = CLIPVisionModelWithProjection.from_pretrained(model_name).eval()
        self.input_image_size = self.image_processor.crop_size["height"]

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, images):
        """Computes CLIP embeddings for a batch of images."""
        images = _resize_bicubic(images, self.input_image_size)
        inputs = self.image_processor(
            images=images,
            do_normalize=True,
            do_center_crop=False,
            do_resize=False,
            do_rescale=False,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        image_embs = self.model(**inputs).image_embeds.cpu()
        image_embs /= torch.linalg.norm(image_embs, axis=-1, keepdims=True)
        return image_embs


class CMMD(KernelInceptionDistance):
    """CMMD: CLIP-based Maximum Mean Discrepancy (MMD) Metric."""

    def __init__(self, subsets=100, subset_size=1000, degree=1, gamma=None, coef=1.0, **kwargs):
        self.clip_feature_extractor = CLIPFeatureExtractor()

        super().__init__(
            feature=self.clip_feature_extractor,
            subsets=subsets,
            subset_size=subset_size,
            degree=degree,
            gamma=gamma,
            coef=coef,
            **kwargs
        )

class GenerativePerplexity(Perplexity):
    """Generative perplexity metric."""

    def __init__(
        self,
        max_length: int,
        gen_ppl_eval_model_name_or_path: str = 'gpt2-large',
        ignore_index: Optional[int] = None,
        **kwargs: dict[str, Any],
    ):
        super().__init__(ignore_index=ignore_index, **kwargs)
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.gen_ppl_eval_model_name_or_path, 
            use_fast=True, 
            add_special_tokens=False
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
            self.tokenizer.pad_token = "<|pad|>"
        self.ignore_index = int(self.tokenizer.pad_token_id) # type: ignore
        
        self.model = AutoModelForCausalLM.from_pretrained(
            gen_ppl_eval_model_name_or_path
        )
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.eval()

        self.tokenizer_kwargs = {
            'return_tensors': 'pt',
            'return_token_type_ids': False,
            'return_attention_mask': True,
            'truncation': True,
            'padding': True,
            'max_length': max_length,
        }

    def _add_special_tokens(self, text: str) -> str:
        return self.tokenizer.bos_token + text + self.tokenizer.eos_token # type: ignore
        
    def update(self, text_samples: List[str]) -> None:
        text_samples = list(map(self._add_special_tokens, text_samples))
        outputs = self.tokenizer(text_samples, **self.tokenizer_kwargs)
        for (tokens_chunk, attn_mask_chunk) in zip(outputs['input_ids'], outputs['attention_mask']): # type: ignore
            logits = self.eval_model(tokens_chunk, attention_mask=attn_mask_chunk)[0]
            total_log_probs, count = _perplexity_update(logits, tokens_chunk, self.ignore_index)
            self.total_log_probs += total_log_probs
            self.count += count


class ClassifierAccuracy(Metric):
    """Classifier accuracy metric."""

    def __init__(
        self, 
        fb: Literal['forward', 'backward'], 
        cls_model: str = 'sentiment-analysis', 
        **kwargs
    ):
        super().__init__(**kwargs)
        self.classifier = pipeline(cls_model)
        if fb == 'forward':
            self.target_lable = 'POSITIVE'
        else:
            self.target_lable = 'NEGATIVE'
        self.register_buffer("predictions", torch.zeros(0))

    def update(self, texts: Union[str, List[str]]):
        """Update the metric with text inputs."""
        # Handle predictions
        predictions = self.classifier(texts)
        predictions = torch.tensor([1 if p['label'] == self.target_lable else 0 for p in predictions]).long() # type: ignore
        self.predictions = torch.cat([self.predictions, predictions], dim=0)

    def compute(self) -> torch.Tensor:
        """Compute the accuracy."""
        return self.predictions.mean()

       
