from typing import List

import torch
from torch import nn
import torch.utils

from dasbm.data import Prior


class DiffusionModel(nn.Module):
    def __init__(
         self,
         input_dim: int,
         num_categories: int, 
         num_timesteps: int,
         timestep_dim: int = 2, 
         layers: List[int] = [128, 128, 128],
     ) -> None: 
        super().__init__()
        self.input_dim = input_dim
        self.num_categories = num_categories
        self.num_timesteps = num_timesteps
        net = []
        ch_prev = input_dim + timestep_dim
        for ch_next in layers:
            net.extend([nn.Linear(ch_prev, ch_next), nn.ReLU()])
            ch_prev = ch_next
        net.append(nn.Linear(ch_prev, num_categories * input_dim))
        self.net = nn.Sequential(*net)
        self.timestep_embedding = nn.Embedding(num_timesteps + 2, timestep_dim)
    
    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x_start_logits = self.net(torch.cat([x_t.float(), self.timestep_embedding(t)], dim=1))
        return x_start_logits.view(-1, self.input_dim, self.num_categories)
         

class D3PM(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_categories: int, 
        num_timesteps: int,
        timestep_dim: int = 2, 
        layers: List[int] = [128, 128, 128],
    ) -> None:
        super().__init__()
        self.model = DiffusionModel(
            input_dim, 
            num_categories,
            num_timesteps,
            timestep_dim,
            layers
        )
        self.num_categories = num_categories
        self.num_timesteps = num_timesteps

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x_one_hot = torch.nn.functional.one_hot(x, self.num_categories) 
        mean = (self.num_categories - 1) / 2
        x = x / mean - 1
        return self.model(x, t) + x_one_hot

    def markov_sample(self, x: torch.Tensor, t: torch.Tensor, prior: Prior):
        r"""Samples from $p(x_{t-1} | x_{t}, \hat{x_{0}})$, where $\hat{x_{0}} \sim m_{\theta}(\hat{x_{0}} | x_{t})$."""
        first_step = (t == 1).long().view((x.shape[0], *[1] * (x.dim() - 1)))
        
        pred_x_end_logits = self(x, t)
        pred_q_posterior_logits = prior.posterior_logits(pred_x_end_logits, x, t, logits=True)
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
