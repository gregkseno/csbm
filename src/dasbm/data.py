from typing import Any, Literal, Optional, Tuple, Union

import numpy as np
import ot
from sklearn.datasets import make_swiss_roll
from PIL import Image
import pandas as pd
import os
import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms

from torchvision import transforms, datasets

from tqdm.auto import tqdm
tqdm.pandas()

from dasbm.utils import broadcast

#########################
#       MARGINALS       #
#########################
class BaseDataset(Dataset):
    dataset: Any

    def __init__(self, *args, **kwargs):
        super().__init__()

    def __getitem__(self, idx):
        return self.dataset[idx]
    
    def __len__(self):
        return len(self.dataset)

    def continuous_to_discrete(self, batch: Union[torch.Tensor, np.ndarray], num_categories: int):
        if isinstance(batch, np.ndarray):
            batch = torch.tensor(batch)
        batch_min = batch.min(dim=0).values
        batch_max = batch.max(dim=0).values

        discrete_batch = torch.zeros_like(batch, dtype=torch.int64)
        for dim, (minn, maxx) in enumerate(zip(batch_min, batch_max)):  
            bin_edges = torch.linspace(minn, maxx, num_categories - 1)
            discrete_batch[:, dim] = torch.bucketize(batch[:, dim].contiguous(), bin_edges)
        return discrete_batch
    
    def repeat(self, n: int, max_len: int):
        self.dataset = self.dataset.repeat((n,) + (1,) * (self.dataset.dim()-1))
        self.dataset = self.dataset[:max_len]
    

class DiscreteGaussianDataset(BaseDataset):
    def __init__(self, n_samples: int, dim: int, num_categories: int = 100, train: bool = True):
        dataset = torch.randn(size=[n_samples, dim])
        dataset = self.continuous_to_discrete(dataset, num_categories)
        if not train:
            dataset[:4] = torch.tensor([[25, 48], [45, 25], [25, 5], [5, 20]])      

        self.dataset = dataset
    
class DiscreteSwissRollDataset(BaseDataset):
    def __init__(self, n_samples: int, noise: float = 0.8, num_categories: int = 100, train: bool = True):
        dataset = make_swiss_roll(
            n_samples=n_samples,
            noise=noise
        )[0][:, [0, 2]]
        dataset = self.continuous_to_discrete(dataset, num_categories)
        if not train:
            dataset[:4] = torch.tensor([[25, 25], [46, 4], [6, 44], [49, 49]])
        self.dataset = dataset      
    
class DiscreteColoredMNISTDataset(BaseDataset):
    def __init__(
        self, 
        target_digit: int, 
        data_dir: str, 
        train: bool = True, 
        img_size: int = 32
    ):
        
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Lambda(lambda image: self._get_random_colored_images(image))
        ])
        
        dataset = datasets.MNIST(data_dir, train=train, transform=transform, download=True)
        dataset = torch.stack(
            [dataset[i][0] for i in range(len(dataset.targets)) if dataset.targets[i] == target_digit],
            dim=0
        )
        dataset = (255 * dataset).to(dtype=torch.int64)
        self.dataset = dataset      

    def _get_random_colored_images(self, image: torch.Tensor):
        hue = 360 * torch.rand(1)
        image_min = 0
        image_diff = (image - image_min) * (hue % 60) / 60
        image_inc = image_diff
        image_dec = image - image_diff
        colored_image = torch.zeros((3, image.shape[1], image.shape[2]))
        H_i = torch.round(hue / 60) % 6 # type: ignore
        
        if H_i == 0:
            colored_image[0] = image
            colored_image[1] = image_inc
            colored_image[2] = image_min
        elif H_i == 1:
            colored_image[0] = image_dec
            colored_image[1] = image
            colored_image[2] = image_min
        elif H_i == 2:
            colored_image[0] = image_min
            colored_image[1] = image
            colored_image[2] = image_inc
        elif H_i == 3:
            colored_image[0] = image_min
            colored_image[1] = image_dec
            colored_image[2] = image
        elif H_i == 4:
            colored_image[0] = image_inc
            colored_image[1] = image_min
            colored_image[2] = image
        elif H_i == 5:
            colored_image[0] = image
            colored_image[1] = image_min
            colored_image[2] = image_dec
        
        return colored_image
    

class CelebaDataset(BaseDataset):
    def __init__(
        self, 
        sex: Literal['male', 'female'], 
        data_dir: str,
        size: int = 128, 
        transform: Optional[None] = None, 
        train: bool = True
    ):
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
        ])
        attrs = pd.read_csv(os.path.join(data_dir, 'celeba', 'list_attr_celeba.csv'))
        if sex == 'male':
            attrs = attrs[attrs['Male'] != -1] # only males
        else:
            attrs = attrs[attrs['Male'] == -1]

        split = pd.read_csv(os.path.join(data_dir, 'celeba', 'list_eval_partition.csv'))
        if train:
            split = split[split['partition'] == 0]
        else:
            split = split[split['partition'] != 0]

        image_names = pd.merge(attrs, split, on=['image_id'], how='inner')
        image_names = image_names['image_id'].tolist()
        self.dataset = [os.path.join(data_dir, 'celeba', 'img_align_celeba', 'img_align_celeba', image) for image in image_names]
        
    def __getitem__(self, index):
        image = Image.open(self.dataset[index])
        image = image.convert('RGB')
        image = self.transform(image)
        return image

    def __len__(self):
        return len(self.dataset)
    
    def repeat(self, n: int, max_len: int):
        self.dataset = self.dataset * n
        self.dataset = self.dataset[:max_len]
        
class CouplingDataset(BaseDataset):
    def __init__(
        self, 
        dataset: BaseDataset, 
        conditional: BaseDataset, 
        type: Literal['independent', 'prior'] = 'independent',
        prior: Optional['Prior'] = None
    ):
        self.type = type
        self.prior = prior

        if len(dataset) != len(conditional):
            max_len = max(len(dataset), len(conditional))
            if len(dataset) < max_len:
                times = (max_len + len(dataset) - 1) // len(dataset)
                dataset.repeat(times, max_len)

            if len(conditional) < max_len:
                times = (max_len + len(conditional) - 1) // len(conditional)
                conditional.repeat(times, max_len)

        self.dataset = dataset
        self.conditional = conditional

    def __getitem__(self, idx):
        if self.type == 'independent':
            x, y = self.dataset[idx], self.conditional[idx]
        elif self.type == 'prior':
            raise NotImplementedError('Only independent coupling is now supported!')
        return x, y

    
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
    
def uniform_prior(
    alpha: float, 
    num_categories: int, 
    num_timesteps: int,
    num_skip_steps: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    p_onestep_mats = torch.tensor([alpha / (num_categories - 1)] * num_categories**2, dtype=torch.float64)
    p_onestep_mats = p_onestep_mats.view(num_categories, num_categories)
    p_onestep_mats -= torch.diag(torch.diag(p_onestep_mats))
    p_onestep_mats += torch.diag(torch.tensor([1 - alpha] * num_categories))
    p_onestep_mats = torch.matrix_power(p_onestep_mats, n=num_skip_steps)

    p_onestep_mats = p_onestep_mats.unsqueeze(0).repeat(num_timesteps + 1, 1, 1)
    p_onestep_mats = torch.cat([torch.eye(num_categories).unsqueeze(0), p_onestep_mats], dim=0)
    p_cum_mats = get_cum_matrices(p_onestep_mats)

    return p_onestep_mats.transpose(1, 2), p_cum_mats

def random_neighbour_prior(
    alpha: float,
    num_categories: int, 
    num_timesteps: int,
    num_skip_steps: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    diag_values = 1 - alpha
    off_diag_values = alpha / 2
    p_onestep_mats = np.zeros([num_categories, num_categories], dtype=np.float64)

    np.fill_diagonal(p_onestep_mats, diag_values)
    p_onestep_mats += np.diag([off_diag_values] * (num_categories - 1), k=1)
    p_onestep_mats += np.diag([off_diag_values] * (num_categories - 1), k=-1)
    p_onestep_mats[0, 1] += off_diag_values
    # p_onestep_mats[1, 0] += off_diag_values
    p_onestep_mats[-1, -2] += off_diag_values
    # p_onestep_mats[-2, -1] += off_diag_values
    p_onestep_mats = np.linalg.matrix_power(p_onestep_mats, n=num_skip_steps)

    p_onestep_mats = torch.from_numpy(p_onestep_mats)
    p_onestep_mats = p_onestep_mats.unsqueeze(0).repeat(num_timesteps + 1, 1, 1)
    p_onestep_mats = torch.cat([torch.eye(num_categories).unsqueeze(0), p_onestep_mats], dim=0)
    p_cum_mats = get_cum_matrices(p_onestep_mats)

    return p_onestep_mats.transpose(1, 2), p_cum_mats

def von_mises_prior(
    alpha: float,
    num_categories: int, 
    num_timesteps: int,
    num_skip_steps: int
):
    angles = np.linspace(0, 2 * np.pi, num_categories, endpoint=False)
    p_onestep_mats = np.zeros((num_categories, num_categories))

    for i, current_angle in enumerate(angles):
        for j, next_angle in enumerate(angles):
            p_onestep_mats[i, j] = np.exp((1 / alpha) * np.cos(next_angle - current_angle))
        p_onestep_mats[i] /= np.sum(p_onestep_mats[i])
    p_onestep_mats = np.linalg.matrix_power(p_onestep_mats, n=num_skip_steps)

    p_onestep_mats = torch.from_numpy(p_onestep_mats)
    p_onestep_mats = p_onestep_mats.unsqueeze(0).repeat(num_timesteps + 1, 1, 1)
    p_onestep_mats = torch.cat([torch.eye(num_categories).unsqueeze(0), p_onestep_mats], dim=0)
    p_cum_mats = get_cum_matrices(p_onestep_mats)

    return p_onestep_mats.transpose(1, 2), p_cum_mats

def gaussian_prior(
    alpha: float,
    num_categories: int, 
    num_timesteps: int, 
    num_skip_steps: int
):
    p_onestep_mats = np.zeros([num_categories, num_categories], dtype=np.float64)
    # indices = np.arange(num_categories)[None, ...]
    # values = -(indices - indices.T)**2
    # p_onestep_mats = values / alpha
    norm_const = -4 * np.arange(
        start=-(num_categories-1), 
        stop=(num_categories+1), 
        step=1, 
        dtype=np.float64
    ) ** 2
    norm_const /= (alpha**2 * (num_categories - 1)**2)
    for i in range(num_categories):
        for j in range(num_categories):
            if i == j:
                continue
            value = -(4 * (i - j)** 2) / (alpha**2 * (num_categories - 1)**2)
            p_onestep_mats[i][j] = np.exp(value) / np.exp(norm_const).sum()

    for i in range(num_categories):
        p_onestep_mats[i][i] = 1 - p_onestep_mats[i].sum() 
    p_onestep_mats = np.linalg.matrix_power(p_onestep_mats, n=num_skip_steps)

    p_onestep_mats = torch.from_numpy(p_onestep_mats) # .softmax(dim=1)
    p_onestep_mats = p_onestep_mats.unsqueeze(0).repeat(num_timesteps + 1, 1, 1)
    p_onestep_mats = torch.cat([torch.eye(num_categories).unsqueeze(0), p_onestep_mats], dim=0)
    p_cum_mats = get_cum_matrices(p_onestep_mats)

    return p_onestep_mats.transpose(1, 2), p_cum_mats


# Cumulative returns with following pattern
# 0         1           2           ...         N           N+1
# 0->0      0->1        0->2        ...         0->N        0->N+1       

# Onestep returns with following pattern
# 0         1           2           ...         N           N+1
# 0->0      0->1        1->2        ...         N-1->N      N->N+1     

# Inherit from nn.Module to automatically do device casting
class Prior(nn.Module):
    def __init__(
        self, 
        num_categories: int,
        num_timesteps: int,
        num_skip_steps: int,
        prior_type: Literal[
            'uniform', 
            'random_neighbour', 
            'gaussian',
            'von_mises'
        ] = 'uniform',
        alpha: Optional[float] = None
    ) -> None:
        super().__init__()
        self.num_categories = num_categories
        self.num_timesteps = num_timesteps
        self.num_skip_steps = num_skip_steps
        self.eps = 1e-6

        if prior_type == 'gaussian' and alpha is not None:
            p_onestep, p_cum = gaussian_prior(alpha, num_categories, num_timesteps, num_skip_steps)
        elif prior_type == 'von_mises' and alpha is not None:
            p_onestep, p_cum = von_mises_prior(alpha, num_categories, num_timesteps, num_skip_steps)
        elif prior_type == 'uniform' and alpha is not None:
            p_onestep, p_cum = uniform_prior(alpha, num_categories, num_timesteps, num_skip_steps)
        elif prior_type == 'random_neighbour' and alpha is not None:
            p_onestep, p_cum = random_neighbour_prior(alpha, num_categories, num_timesteps, num_skip_steps)
        else:
            raise NotImplementedError(f'Got unknown prior: {prior_type} or alpha is None!')
        self.register_buffer("p_onestep", p_onestep)
        self.register_buffer("p_cum", p_cum)
        
    def probs(
        self, 
        mat_type: Literal['onestep', 'cumulative'], 
        t: torch.Tensor, 
        *,
        x_start: Optional[torch.Tensor] = None,
        x_end: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Extracts probs from transition matrix."""
        p_mat = self.p_onestep if mat_type == 'onestep' else self.p_cum
        
        if x_start is not None and x_end is not None:
            t = broadcast(t, x_start.dim() - 1)
            return p_mat[t, x_start, x_end].unsqueeze(-1)
        if x_start is not None and x_end is None:
            t = broadcast(t, x_start.dim() - 1)
            return p_mat[t, x_start]
        if x_start is None and x_end is not None:
            t = broadcast(t, x_end.dim() - 1)
            return p_mat[t, :, x_end]
        raise ValueError('x_start and x_end cannot be None both!')

    def sample_bridge(self, x_start: torch.Tensor, x_end: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        r"""Samples from bridge $p(x_{t} | x_{0}, x_{1})$."""
        p_start_t = self.probs('cumulative', t, x_start=x_start)
        p_t_end = self.probs('cumulative', self.num_timesteps + 1 - t, x_end=x_end) # TODO: убедиться что это корректно
        log_probs = torch.log(p_start_t + self.eps) + torch.log(p_t_end + self.eps) # - torch.log(p_start_end + self.eps)
        
        noise = torch.rand_like(log_probs)
        noise = torch.clamp(noise, min=torch.finfo(noise.dtype).tiny, max=1.)
        gumbel_noise = -torch.log(-torch.log(noise))
        x_t = torch.argmax(log_probs + gumbel_noise, dim=-1)
        # probs = log_probs.softmax(dim=-1).view(-1, self.num_categories)
        # x_t = probs.multinomial(num_samples=1).view(x_start.shape)

        is_final_step = broadcast(t, x_start.dim() - 1) == self.num_timesteps + 1
        x_t = torch.where(is_final_step, x_end, x_t)
        return x_t
    
    def posterior_logits(
        self, 
        x_start: torch.Tensor, 
        x_t: torch.Tensor, 
        t: torch.Tensor, 
        logits: bool = False
    ) -> torch.Tensor:
        r"""Calculates logits of $p(x_{t-1} | x_{t}, x_{0})$.""" 
        if not logits:
            x_start_logits = torch.log(torch.nn.functional.one_hot(x_start, self.num_categories) + self.eps)
        else:
            x_start_logits = x_start.clone()
        assert x_start_logits.shape == x_t.shape + (self.num_categories,), f"x_start_logits.shape: {x_start_logits.shape}, x_t.shape: {x_t.shape}"
        
        # fact1 is "guess of x_{t}" from x_{t-1}
        fact1 = self.probs('onestep', t, x_start=x_t)

        # fact2 is "guess of x_{t-1}" from x_{0}
        x_start_probs = x_start_logits.softmax(dim=-1)  # bs, ..., num_categories
        fact2 = torch.einsum("b...c,bcd->b...d", x_start_probs, self.p_cum[t - 1].to(dtype=x_start_probs.dtype))
        p_posterior_logits = torch.log(fact1 + self.eps) + torch.log(fact2 + self.eps)
        
        # Use `torch.where` because when `t == 1` x_start_logits are actually x_0 already
        is_first_step = broadcast(t, x_t.dim()) == 1
        p_posterior_logits = torch.where(is_first_step, x_start_logits, p_posterior_logits)
        return p_posterior_logits

