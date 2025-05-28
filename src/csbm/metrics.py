import os
from typing import Any, List, Literal, Optional, Union
from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F

from torchmetrics import Metric
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.text import EditDistance, BLEUScore
from torchmetrics.regression import MeanSquaredError as MSE
from torchmetrics.classification import MulticlassHammingDistance as HammingDistance
from torch.hub import download_url_to_file

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


class FID(FrechetInceptionDistance):
    def __init__(
        self,
        feature: Union[int, nn.Module] = 2048,
        reset_real_features: bool = False,
        normalize: bool = True,
        input_img_size: tuple[int, int, int] = (3, 299, 299),
        feature_extractor_weights_path: Optional[str] = 'checkpoints/fid_weights.ckpt',
        **kwargs: Any,
    ) -> None:
        if feature_extractor_weights_path is not None:
            if not os.path.exists(feature_extractor_weights_path):
                os.makedirs(os.path.dirname(feature_extractor_weights_path), exist_ok=True)
                print(f"Downloading FID weights to {feature_extractor_weights_path}...")
                download_url_to_file(
                    url=FID_WEIGHTS_URL, 
                    dst=feature_extractor_weights_path,
                    progress=True
                )
        super().__init__(
            feature=feature,
            reset_real_features=reset_real_features,
            normalize=normalize,
            input_img_size=input_img_size,
            feature_extractor_weights_path=feature_extractor_weights_path,
            **kwargs,
        )


class CMMD(Metric):
    def __init__(
        self,
        reset_real_features: bool = False,
        normalize: bool = True,
        embedding_extractor_model: str = CLIP_MODEL_NAME,
    ) -> None:
        super().__init__()
        self.image_processor = CLIPImageProcessor.from_pretrained(embedding_extractor_model)
        self.model = CLIPVisionModelWithProjection.from_pretrained(embedding_extractor_model).eval()
        self.input_image_size = self.image_processor.crop_size["height"]
        self.normalize = normalize
        self.reset_real_features = reset_real_features
        self.is_resetted = False
    
        self.add_state("real_images", default=torch.zeros(0))
        self.add_state("fake_images", default=torch.zeros(0))

    def _mmd(
        self, 
        x: torch.Tensor, 
        y: torch.Tensor, 
        sigma: int = 
        10, scale: int = 1000
    ) -> torch.Tensor:

        x_sqnorms = torch.diag(torch.matmul(x, x.T))
        y_sqnorms = torch.diag(torch.matmul(y, y.T))

        gamma = 1 / (2 * sigma**2)
        k_xx = torch.mean(
            torch.exp(-gamma * (-2 * torch.matmul(x, x.T) + torch.unsqueeze(x_sqnorms, 1) + torch.unsqueeze(x_sqnorms, 0)))
        )
        k_xy = torch.mean(
            torch.exp(-gamma * (-2 * torch.matmul(x, y.T) + torch.unsqueeze(x_sqnorms, 1) + torch.unsqueeze(y_sqnorms, 0)))
        )
        k_yy = torch.mean(
            torch.exp(-gamma * (-2 * torch.matmul(y, y.T) + torch.unsqueeze(y_sqnorms, 1) + torch.unsqueeze(y_sqnorms, 0)))
        )

        return scale * (k_xx + k_yy - 2 * k_xy)

    def update(self, imgs: torch.Tensor, real: bool) -> None:
        if not self.reset_real_features and self.is_resetted and real:
            # We dont want to update the real features 
            # after reset if reset_real_features is False
            return
        imgs = (imgs / 255).float() if not self.normalize else imgs
        imgs = F.interpolate(
            imgs, size=(self.input_image_size, self.input_image_size), mode="bicubic"
        )
        inputs = self.image_processor(
            images=imgs,
            do_normalize=True,
            do_center_crop=False,
            do_resize=False,
            do_rescale=False,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        image_embs = self.model(**inputs).image_embeds
        image_embs /= torch.linalg.norm(image_embs, axis=-1, keepdims=True)
        if real:
            self.real_images = torch.cat([self.real_images, image_embs], dim=0)
        else:
            self.fake_images = torch.cat([self.fake_images, image_embs], dim=0)

    def compute(self) -> torch.Tensor:
        return self._mmd(self.real_images, self.fake_images)
    
    def reset(self) -> None:
        self.is_resetted = True
        if not self.reset_real_features:
            real_images = deepcopy(self.real_images)
            super().reset()
            self.real_images = real_images
        else:
            super().reset()    

    @property
    def device(self):
        return next(self.parameters()).device

class GenerativeNLL(Metric):
    """Generative negative log-likelihood."""

    def __init__(
        self,
        gen_ppl_model: str = 'gpt2-large',
        stride: int = 1,
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
        self.stride = stride
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

        for begin_loc in range(0, seq_len, self.stride):
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
        return self.nlls.mean() # type: ignore
        
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


       
