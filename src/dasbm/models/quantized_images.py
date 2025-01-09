from typing import Any, Optional
import torch
import torch.nn as nn

from dasbm.data import Prior
from dasbm.vq_diffusion.modeling.codecs.image_codec.taming_gumbel_vqvae import TamingFFHQVQVAE
from dasbm.vq_diffusion.modeling.transformers.transformer_utils import UnCondition2ImageTransformer


class VectorQuantizer(TamingFFHQVQVAE):
    def __init__(
        self,
        config_path: str,
        ckpt_path: str,
        latent_size=16,
        num_categories: int = 1024,
    ) -> None:
        super().__init__(
            trainable=False,
            token_shape=[latent_size, latent_size],
            config_path=config_path,
            ckpt_path=ckpt_path,
            num_tokens=num_categories,
            quantize_number=0,
            mapping_path=None
        )

class LatentD3PM(nn.Module):
    def __init__(
        self, 
        latent_size=16,
        num_categories: int = 1024,
        num_timesteps: int = 100,
        hidden_dim: int = 512,
        num_channels: int = 4,
        num_layers: int = 24,
        num_att_heads: int = 16,
        dropout: float = .0,
    ) -> None:
        super().__init__()
        content_emb_config = {
            'num_embed': num_categories, # total number of embeddings (vocabulary size) i.e. num_categories
            'spatial_size': int(num_categories ** 0.5), # for positional encoding it is image like (squared)
            'embed_dim': hidden_dim, # size of one embedding
            'trainable': True,
            'pos_emb_type': 'embedding',
        }

        self.model = UnCondition2ImageTransformer(
            n_layer=num_layers,
            n_embd=hidden_dim, # embed beacuse of time encoding
            n_head=num_att_heads,
            content_seq_len=latent_size*latent_size,
            attn_pdrop=dropout,
            resid_pdrop=dropout,
            mlp_hidden_times=num_channels,
            block_activate='GELU2',
            attn_type='self',
            content_spatial_size=[latent_size, latent_size],
            diffusion_step=num_timesteps,
            timestep_type='adalayernorm',
            content_emb_config=content_emb_config,
            mlp_type='conv_mlp',
        )

        self.num_categories = num_categories
        self.num_timesteps = num_timesteps

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # x_one_hot = F.one_hot(x, self.num_categories) 
        # mean = (self.num_categories - 1) / 2
        # x = x / mean - 1
        return self.model(x, t)  # + x_one_hot

    def markov_sample(self, x: torch.Tensor, t: torch.Tensor, prior: Prior):
        r"""Samples from $p(x_{t-1} | x_{t}, \hat{x_{0}})$, where $\hat{x_{0}} \sim m_{\theta}(\hat{x_{0}} | x_{t})$."""
        first_step = (t == 1).long().view((x.shape[0], *[1] * (x.dim() - 1)))
        
        pred_x_start_logits = self(x, t)
        pred_q_posterior_logits = prior.posterior_logits(pred_x_start_logits, x, t, logits=True)
        noise = torch.rand_like(pred_q_posterior_logits)
        noise = torch.clamp(noise, min=torch.finfo(noise.dtype).tiny, max=1.)
        gumbel_noise = -torch.log(-torch.log(noise))
        random_samples = torch.argmax(pred_q_posterior_logits + gumbel_noise, dim=-1)
        # probs = pred_q_posterior_logits.softmax(dim=-1).view(-1, self.num_categories)
        # random_samples = probs.multinomial(num_samples=1).view(x.shape)
        
        # No noise when t == 1
        # NOTE: for t=1 this just "samples" from the argmax
        #   as opposed to "sampling" from the mean in the gaussian case.

        argmax_samples = pred_q_posterior_logits.argmax(dim=-1)
        samples = first_step * argmax_samples + (1 - first_step) * random_samples
        return samples
        
    @torch.no_grad()
    def sample(self, x: torch.Tensor, prior: Prior) -> torch.Tensor:
        for t in reversed(range(1, self.num_timesteps + 2)):
            t = torch.tensor([t] * x.shape[0], device=self.device)
            x = self.markov_sample(x, t, prior)
        return x
    
    @torch.no_grad()
    def sample_trajectory(self, x: torch.Tensor, prior: Prior) -> torch.Tensor:
        trajectory = [x]
        for t in reversed(range(1, self.num_timesteps + 2)):
            t = torch.tensor([t] * x.shape[0], device=self.device)
            x = self.markov_sample(x, t, prior)
            trajectory.append(x)
        trajectory = torch.stack(trajectory, dim=0)
        return trajectory
    
    @property
    def device(self):
        return next(self.parameters()).device