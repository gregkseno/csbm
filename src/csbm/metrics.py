import os
from typing import Any, List, Literal, Optional, Union

import torch
from torch import nn
from torch.nn import functional as F

from torchmetrics import Metric
from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.image.fid import FrechetInceptionDistance as FID
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS
from torchmetrics.text import EditDistance, BLEUScore
from torchmetrics.regression import MeanSquaredError as MSE
from torchmetrics.classification import MulticlassHammingDistance as HammingDistance
from torchmetrics.functional.text.perplexity import _perplexity_update

from transformers import (
    CLIPImageProcessor, 
    CLIPVisionModelWithProjection, 
    AutoModelForCausalLM, 
    AutoTokenizer,
    DistilBertTokenizer, 
    DistilBertForSequenceClassification
)
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

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

class GenerativeNLL(Metric):
    """Generative negative log-likelihood."""

    def __init__(
        self,
        gen_ppl_model: str = 'gpt2-large',
        context_len: int = 1,
        **kwargs
    ):
        super().__init__(**kwargs)
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'

        self.tokenizer = AutoTokenizer.from_pretrained(
            gen_ppl_model, 
            use_fast=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            gen_ppl_model
        )
        self.model.eval()
        self.context_len = context_len
        self.max_len = self.model.config.n_positions

        self.tokenizer_kwargs = {
            'return_tensors': 'pt',
            'padding': True,
        }
        self.add_state("nlls", default=torch.zeros(0))
        
    def update(self, text_samples: List[str]) -> None:
        encodings = self.tokenizer(text_samples, **self.tokenizer_kwargs)
        seq_len = encodings.input_ids.shape[1]
        prev_end_loc = 0

        for begin_loc in range(0, seq_len, self.context_len):
            end_loc = min(begin_loc + self.max_len, seq_len)
            trg_len = end_loc - prev_end_loc
            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(self.device)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100
            target_ids[target_ids == self.tokenizer.pad_token_id] = -100

            with torch.no_grad():
                output: CausalLMOutputWithCrossAttentions = self.model(
                    input_ids, labels=target_ids
                )
            self.nlls = torch.cat([self.nlls, output.loss.unsqueeze(0)], dim=0) # type: ignore

            prev_end_loc = end_loc
            if end_loc == seq_len:
                break

    def compute(self) -> torch.Tensor:
        return self.nlls.sum() # type: ignore
        
    @property
    def device(self):
        return next(self.parameters()).device


class ClassifierAccuracy(Metric):
    """Classifier accuracy metric."""

    def __init__(
        self, 
        fb: Literal['forward', 'backward'], 
        cls_model: str = 'distilbert-base-uncased-finetuned-sst-2-english',
        **kwargs
    ):
        super().__init__(**kwargs)
        self.tokenizer = DistilBertTokenizer.from_pretrained(cls_model)
        self.model = DistilBertForSequenceClassification.from_pretrained(cls_model)
        self.tokenizer_kwargs = {
            'return_tensors': 'pt',
            'padding': True,
        }
        if fb == 'forward':
            self.target_class = 'positive'
        else:
            self.target_class = 'negative'
        self.add_state("predictions", default=torch.zeros(0))

    def update(self, texts: Union[str, List[str]]):
        """Update the metric with text inputs."""
        inputs = self.tokenizer(texts, **self.tokenizer_kwargs)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            logits: torch.Tensor = self.model(**inputs).logits
        predictions = logits.argmax(dim=-1)
        self.predictions = torch.cat([self.predictions, predictions], dim=0)


    def compute(self) -> torch.Tensor:
        if self.target_class == 'positive':
            return self.predictions.mean()
        else:
            return 1 - self.predictions.mean()
    
    @property
    def device(self):
        return next(self.parameters()).device


       
