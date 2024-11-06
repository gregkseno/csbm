from typing import Literal, Optional, Tuple
import os

import numpy as np
from sklearn.datasets import make_swiss_roll
import torch
from torch import nn
from rdkit import Chem
from torch_geometric.utils import to_dense_adj, to_dense_batch, remove_self_loops, one_hot
from torch_geometric.data import Batch, Data, InMemoryDataset

import logging
from tqdm.auto import tqdm
tqdm.pandas()

from dasbm.utils import broadcast

#########################
#       MARGINALS       #
#########################
class Sampler:
    num_categories: int    

    def __init__(self):
        pass

    def __call__(self, batch_size: int) -> torch.Tensor:
        return self.sample(batch_size)

    def sample(self, batch_size: int) -> torch.Tensor:
        raise NotImplementedError('Abstract Class')
    
    def continuous_to_discrete(self, batch: torch.Tensor | np.ndarray):
        if isinstance(batch, np.ndarray):
            batch = torch.tensor(batch)
        bin_edges = torch.linspace(batch.min(), batch.max(), self.num_categories - 1)
        discrete_batch = torch.bucketize(batch, bin_edges)
        return discrete_batch
    
    def __str__(self):
        return self.__class__.__name__
    

class DiscreteGaussianSampler(Sampler):
    def __init__(self, dim: int, num_categories: int = 100):
        super().__init__()        
        self.dim = dim
        self.num_categories = num_categories

    def sample(self, batch_size: int) -> torch.Tensor:
        batch = torch.randn(size=[batch_size, self.dim])
        return self.continuous_to_discrete(batch)
    
class DiscreteSwissRollSampler(Sampler):
    def __init__(self, noise: float = 0.8, num_categories: int = 100):
        super().__init__()
        self.num_categories = num_categories
        self.noise = noise
        
    def sample(self, batch_size: int) -> torch.Tensor:
        batch = make_swiss_roll(
            n_samples=batch_size,
            noise=self.noise
        )[0][:, [0, 2]] / 7.5
        return self.continuous_to_discrete(batch)
    
def encode_no_edge(edges: torch.Tensor):
    assert len(edges.shape) == 4
    if edges.shape[-1] == 0:
        return edges
    no_edge = torch.sum(edges, dim=3) == 0
    first_elt = edges[:, :, :, 0]
    first_elt[no_edge] = 1
    edges[:, :, :, 0] = first_elt
    diag = torch.eye(edges.shape[1], dtype=torch.bool).unsqueeze(0).expand(edges.shape[0], -1, -1)
    edges[diag] = 0
    return edges
    
class MoluculeData(Data):
    x: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    y: torch.Tensor

class MoleculeBatch(Batch):
    x: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    y: torch.Tensor
    batch: torch.Tensor
    max_num_nodes: int

    def __init__(
        self,
        nodes: torch.Tensor, 
        edges: torch.Tensor, 
        features: torch.Tensor, 
        mask: torch.Tensor
    ):
        # TODO: from_dense implementation
        pass

    def to_dense(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        nodes, mask = to_dense_batch(
            x=self.x, 
            batch=self.batch, 
            max_num_nodes=self.max_num_nodes
        )
        edge_index, edge_attr = remove_self_loops(
            edge_index=self.edge_index, 
            edge_attr=self.edge_attr
        )
        edges = to_dense_adj(
            edge_index=edge_index, 
            batch=self.batch, 
            edge_attr=edge_attr, 
            max_num_nodes=self.max_num_nodes
        )
        edges = encode_no_edge(edges)
        return nodes.float(), edges.float(), mask
    
class MoleculeDatasetSampler(Sampler):
    def __init__(self, dataset: InMemoryDataset, mean: float, std: float):
        super().__init__()
        self.logger = logging.getLogger('DatasetSampler')
        self.dataset = dataset
        # TODO: mean, std filtration
        self.last_index = 0

    def sample(self, batch_size: int) -> MoleculeBatch:
        assert batch_size <= len(self.dataset), f'Bach size: {batch_size} is larger than length of dataset: {len(self.dataset)}'
        if self.last_index + batch_size >= len(self.dataset):
            self.last_index = 0
            self.dataset = self.dataset.shuffle() # type: ignore
        batch_indices = torch.arange(self.last_index, self.last_index + batch_size)         
        batch = MoleculeBatch.from_data_list(self.dataset[batch_indices]) # type: ignore
        self.last_index += batch_size
        return batch

    
#########################
#         Priors        #
#########################
def get_cum_matrices(onestep_matrices: torch.Tensor) -> torch.Tensor:
    matrix = onestep_matrices[0]
    cum_matrices = [matrix]
    for timestep in range(1, onestep_matrices.shape[0]):
        matrix = matrix @ onestep_matrices[timestep]
        cum_matrices.append(matrix)
    cum_matrices = torch.stack(cum_matrices, dim=0)

    assert onestep_matrices.shape == cum_matrices.shape, f'Wrong shape!'
    return cum_matrices
    
def random_jump_prior(
    alpha: float, 
    num_categories: int, 
    num_timesteps: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    q_onestep_mats = torch.tensor([alpha / (num_categories - 1)] * num_categories**2, dtype=torch.float64)
    q_onestep_mats = q_onestep_mats.reshape(num_categories, num_categories)
    q_onestep_mats -= torch.diag(torch.diag(q_onestep_mats))
    q_onestep_mats += torch.diag(torch.tensor([1 - alpha] * num_categories))
    q_onestep_mats = q_onestep_mats.unsqueeze(0).repeat(num_timesteps + 2, 1, 1)

    q_cum_mats = get_cum_matrices(q_onestep_mats)

    return q_onestep_mats.transpose(1, 2), q_cum_mats

def d3pm_prior(
    num_categories: int, 
    num_timesteps: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    
    q_onestep_mats = []
    for timestep in range(0, num_timesteps + 2):
        beta = 1 / (num_timesteps - timestep + 2)
        mat = torch.ones(num_categories, num_categories) * beta / num_categories
        mat.diagonal().fill_(1 - (num_categories - 1) * beta / num_categories)
        q_onestep_mats.append(mat)
    q_onestep_mats = torch.stack(q_onestep_mats, dim=0)

    q_cum_mats = get_cum_matrices(q_onestep_mats)

    return q_onestep_mats.transpose(1, 2), q_cum_mats

def random_neighbour_prior(
    alpha: float,
    num_categories: int, 
    num_timesteps: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    q_onestep_mats = torch.diag(torch.tensor([1 - alpha] * num_categories))
    q_onestep_mats += torch.diag(torch.tensor([alpha / 2] * (num_categories - 1)), diagonal=1)
    q_onestep_mats += torch.diag(torch.tensor([alpha / 2] * (num_categories - 1)), diagonal=-1)
    q_onestep_mats = q_onestep_mats.unsqueeze(0).repeat(num_timesteps + 2, 1, 1)

    q_cum_mats = get_cum_matrices(q_onestep_mats)

    return q_onestep_mats.transpose(1, 2), q_cum_mats

# Cumulative returns with following pattern
# 0         1           2           3           ...
# 0->1      0->2        0->3        0->4        ...

# Onestep returns with following pattern
# 0         1           2           3           ...
# 0->1      1->2        2->3        3->4        ...

# Inherit from nn.Module to automatically do device casting
class Prior(nn.Module):
    def __init__(
        self, 
        alpha: Optional[float],
        num_categories: int,
        num_timesteps: int,
        prior_type: Literal['random_jump', 'random_neighbour', 'd3pm'] = 'random_jump'
    ) -> None:
        super().__init__()
        self.num_categories = num_categories
        self.num_timesteps = num_timesteps
        self.eps = 1e-6

        if prior_type == 'd3pm':   
            q_onestep, q_cum = d3pm_prior(num_categories, num_timesteps)
        elif prior_type == 'random_jump' and alpha is not None:
            q_onestep, q_cum = random_jump_prior(alpha, num_categories, num_timesteps)
        elif prior_type == 'random_neighbour' and alpha is not None:
            q_onestep, q_cum = random_neighbour_prior(alpha, num_categories, num_timesteps)
        else:
            raise NotImplementedError(f'Got unknown prior: {prior_type} or alpha is None!')
        self.register_buffer("q_onestep", q_onestep)
        self.register_buffer("q_cum", q_cum)
        
    def _qt(self, q_mat: torch.Tensor, t: torch.Tensor, x: torch.Tensor):
        """Extracts transition matrix."""
        t = broadcast(t, x.dim() - 1)
        return q_mat[t, x]

    def sample(self, x_start: torch.Tensor, t: torch.Tensor):
        """Samples from $p(x_{t} | x_{0})$."""
        # noise = torch.rand((*x_start.shape, self.num_categories), device=x_start.device)
        # noise = torch.clip(noise, self.eps, 1.0)
        # gumbel_noise = -torch.log(-torch.log(noise))
        # logits = torch.log() # do -1 to convert from time to index to match pattern from q_cum
        # torch.argmax(logits + gumbel_noise, dim=-1)
        batch_size, dim = x_start.shape[:2]
        probs = self._qt(self.q_cum, t - 1, x_start)
        probs = probs.reshape(batch_size * dim, -1)
        return probs.multinomial(num_samples=1).reshape(x_start.shape)
    
    def q_posterior_logits(self, x_start: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor):
        """Calculates logits of $p(x_{t-1} | x_{t}, x_{0})$.""" 

        # if x_end is integer, we convert it to one-hot.
        if not torch.is_floating_point(x_start) and not torch.is_complex(x_start):
            x_start_logits = torch.log(torch.nn.functional.one_hot(x_start, self.num_categories) + self.eps)
        else:
            x_start_logits = x_start.clone()
        assert x_start_logits.shape == x_t.shape + (self.num_categories,), print( f"x_start_logits.shape: {x_start_logits.shape}, x_t.shape: {x_t.shape}")
        
        # fact1 is "guess of x_{t}" from x_{t-1}
        fact1 = self._qt(self.q_onestep, t - 1, x_t)

        # fact2 is "guess of x_{t-1}" from x_{0}
        x_start_probs = torch.softmax(x_start_logits, dim=-1)  # bs, ..., num_categories
        # q_mat = self.q_cum[t - 2] # .to(dtype=x_start_probs.dtype) 
        # fact2 = torch.einsum("b...c,bcd->b...d", x_start_probs, q_mat) # bs, num_categories, num_categories
        fact2 = x_start_probs @ self.q_cum[t - 2].to(dtype=x_start_probs.dtype) 
        q_posterior_logits = torch.log(fact1 + self.eps) + torch.log(fact2 + self.eps)

        # Use `torch.where` because when `t == 1` x_start_logits are actually x_0 already
        return torch.where(broadcast(t, x_t.dim()) == 1, x_start_logits, q_posterior_logits)
