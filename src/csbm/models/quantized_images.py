from omegaconf import OmegaConf
import torch
import torch.nn as nn

from csbm.data import Prior
from csbm.vq_diffusion.modeling.codecs.image_codec.taming_gumbel_vqvae import TamingVQVAE
from csbm.vq_diffusion.modeling.transformers.transformer_utils import UnCondition2ImageTransformer
from csbm.vq_diffusion.taming.models.vqgan import VQModel


class Codec(nn.Module):
    def __init__(
        self,
        config_path: str,
        ckpt_path: str,
    ) -> None:
        super().__init__()
        self.config = OmegaConf.load(config_path).model.params
        self.model = VQModel(**self.config)
        state_dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        self.model.load_state_dict(state_dict, strict=False)
        self.model = self.model.eval()
        self.centroids = self.model.quantize.embedding.weight.data

    @torch.no_grad()
    def encode_to_cats(self, images: torch.Tensor) -> torch.Tensor:
        images = 2 * images - 1
        _, _, (_, _, cats) = self.model.encode(images)
        cats = cats.reshape(images.shape[0], -1)
        return cats.long()
    
    @torch.no_grad()
    def decode_to_image(self, cats: torch.Tensor) -> torch.Tensor:
        shape = (
            cats.shape[0], 
            int(self.config.embed_dim ** 0.5), 
            int(self.config.embed_dim ** 0.5), 
            int(self.config.ddconfig.z_channels)
        )
        z_q = self.model.quantize.get_codebook_entry(cats, shape)
        images = self.model.decode(z_q)
        images = torch.clamp(images, -1., 1.)
        images = (images + 1.) / 2.
        return images
    
    @property
    def device(self):
        return next(self.parameters()).device

class LatentD3PM(nn.Module):
    def __init__(
        self, 
        input_dim: int,
        num_categories: int,
        num_timesteps: int,
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
            content_seq_len=input_dim*input_dim,
            attn_pdrop=dropout,
            resid_pdrop=dropout,
            mlp_hidden_times=num_channels,
            block_activate='GELU2',
            attn_type='self',
            content_spatial_size=[input_dim, input_dim],
            diffusion_step=num_timesteps + 2,
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
        first_step = (t == 1).view((x.shape[0], *[1] * (x.dim() - 1))).to(dtype=x.dtype)
        
        pred_x_start_logits = self(x, t)
        pred_q_posterior_logits = prior.posterior_logits(pred_x_start_logits, x, t, logits=True)
        noise = torch.rand_like(pred_q_posterior_logits, dtype=x.dtype)
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
            t = torch.tensor([t] * x.shape[0], device=self.device, dtype=x.dtype)
            x = self.markov_sample(x, t, prior)
        return x
    
    @torch.no_grad()
    def sample_trajectory(self, x: torch.Tensor, prior: Prior) -> torch.Tensor:
        trajectory = [x]
        for t in reversed(range(1, self.num_timesteps + 2)):
            t = torch.tensor([t] * x.shape[0], device=self.device, dtype=x.dtype)
            x = self.markov_sample(x, t, prior)
            trajectory.append(x)
        trajectory = torch.stack(trajectory, dim=0)
        return trajectory
    
    @property
    def device(self):
        return next(self.parameters()).device