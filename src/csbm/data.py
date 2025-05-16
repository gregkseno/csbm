import sys
from typing import Any, List, Literal, Optional, Tuple, Union

import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import softmax
from sklearn.datasets import make_swiss_roll
from PIL import Image
import pandas as pd
import os
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from transformers import PreTrainedTokenizerFast
import json

from tqdm.auto import tqdm
tqdm.pandas()

from csbm.utils import broadcast, convert_to_numpy, convert_to_torch

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

    def continuous_to_discrete(
        self, 
        batch: Union[torch.Tensor, np.ndarray], 
        num_categories: int,
        quantize_range: Optional[Tuple[Union[int, float], Union[int, float]]] = None
    ):
        if isinstance(batch, np.ndarray):
            batch = torch.tensor(batch)
        if quantize_range is None:
            quantize_range = (-3, 3)
        bin_edges = torch.linspace(
            quantize_range[0], 
            quantize_range[1], 
            num_categories - 1
        )
        discrete_batch = torch.bucketize(batch, bin_edges)
        return discrete_batch
    
    def repeat(self, n: int, max_len: int):
        self.dataset = self.dataset.repeat((n,) + (1,) * (self.dataset.dim()-1))
        self.dataset = self.dataset[:max_len]
    

class DiscreteGaussianDataset(BaseDataset):
    def __init__(self, n_samples: int, dim: int, num_categories: int = 100, train: bool = True):
        dataset = torch.randn(size=[n_samples, dim])
        if not train:
            dataset[:4] = torch.tensor([[0.0, 0.0], [1.75, -1.75], [-1.5, 1.5], [2, 2]])
            
        dataset = self.continuous_to_discrete(dataset, num_categories)
        self.dataset = dataset
    
class DiscreteSwissRollDataset(BaseDataset):
    def __init__(self, n_samples: int, noise: float = 0.8, num_categories: int = 100, train: bool = True):
        dataset = make_swiss_roll(
            n_samples=n_samples,
            noise=noise
        )[0][:, [0, 2]]  / 7.5
        # dataset = (dataset - dataset.mean()) / dataset.std()
        if not train:
            dataset[:4] = torch.tensor([[0.0, 0.0], [1.75, -1.75], [-1.5, 1.5], [2, 2]])
        dataset = self.continuous_to_discrete(dataset, num_categories)
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


class ImageFolder(datasets.ImageFolder):
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, _ = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, path

class CelebaDataset(BaseDataset):
    transform: Optional[transforms.Compose] = None
    
    def __init__(
        self, 
        sex: Literal['male', 'female', 'both'], 
        data_dir: str,
        size: Optional[int] = None, 
        train: bool = True,
        split: int | float = 162771, # from original dataset
        use_quantized: bool = True,
        return_names: bool = False
    ):
        self.train = train
        self.use_quantized = use_quantized
        self.size = size
        self.return_names = return_names
        self.data_dir= data_dir

        subset = pd.read_csv(os.path.join(data_dir, 'celeba', 'list_attr_celeba.csv'))

        if isinstance(split, int): 
            # this logic mathches setup of previously trained models
            subset = subset.iloc[:split] if train else subset.iloc[split:]
            if sex == 'male':
                subset = subset[subset['Male'] != -1]
            elif sex == 'female':
                subset = subset[subset['Male'] == -1]
            else:
                subset = subset
        else:
            # this logic mathches asbm setup
            male_subset = subset[subset['Male'] != -1]
            female_subset = subset[subset['Male'] == -1]
            male_split_index, female_split_index = int(len(male_subset) * split), int(len(female_subset) * split)
            
            male_subset = male_subset[:male_split_index] if train else male_subset.iloc[male_split_index:]
            female_subset = female_subset[:female_split_index] if train else female_subset.iloc[female_split_index:]

            if sex == 'male':
                subset = male_subset
            elif sex == 'female':
                subset = female_subset
            else:
                subset = pd.concat([male_subset, female_subset], ignore_index=True)
                subset = subset.sort_values(by='image_id').reset_index(drop=True)


        if use_quantized:
            sub_folder = 'quantized'
            subset['image_id'] = subset['image_id'].str.removesuffix('.jpg') + '.npy'
        else:
            sub_folder = 'raw'

        self.image_names = subset['image_id']
        self.dataset = [os.path.join(data_dir, 'celeba', 'img_align_celeba', sub_folder, image) for image in self.image_names.tolist()]

    def __getitem__(self, index):
        if self.train and self.use_quantized:
            image = torch.from_numpy(np.load(self.dataset[index]))
        else:
            transform = transforms.Compose([
                transforms.Resize((self.size, self.size)),
                transforms.ToTensor(),
            ])
            image = Image.open(self.dataset[index])
            image = image.convert('RGB')
            image = transform(image)

        if self.return_names:
           return image, self.dataset[index].split('/')[-1]
        return image

    def __len__(self):
        return len(self.dataset)
    
    def get_by_filename(self, index):
        transform = transforms.Compose([
                transforms.Resize((self.size, self.size)),
                transforms.ToTensor(),
        ])
        # image = self.image_names[self.image_names == index].item()
        image = Image.open(os.path.join(self.data_dir, 'celeba', 'img_align_celeba', 'raw', index))
        image = image.convert('RGB')
        image = transform(image)
        return image
    
    def repeat(self, n: int, max_len: int):
        self.dataset = self.dataset * n
        self.dataset = self.dataset[:max_len]

    @staticmethod
    def quantize_train(
        model: nn.Module, 
        data_dir: str,
        size: int = 128, 
        batch_size: int = 32,
    ):
        load_transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
        ])

        data_dir = os.path.join(data_dir, 'celeba', 'img_align_celeba')
        save_path = os.path.join(data_dir, 'quantized')
        dataset = ImageFolder(data_dir, transform=load_transform, allow_empty=True) # allow_eppty because it will be handled in next line
        if 'quantized' in dataset.classes:
            raise FileExistsError('Folder with quantized images already exists!')
        else:
            os.makedirs(save_path, exist_ok=True)

        dataloader = DataLoader(dataset, batch_size=batch_size)
        for images, image_paths in tqdm(dataloader, file=sys.stdout):
            images = images.to(model.device)
            encoded_images = model.encode_to_cats(images).cpu().detach().numpy()
            for encoded_image, image_path in zip(encoded_images, image_paths):
                file_name = image_path.split('/')[-1].split('.')[0]
                image_path = os.path.join(save_path, file_name)
                np.save(image_path, encoded_image)  


class AFHQDataset(BaseDataset):
    transform: Optional[transforms.Compose] = None
    
    def __init__(
        self, 
        animal_type: Literal['cat', 'wild', 'dog'], 
        data_dir: str,
        size: Optional[int] = None, 
        train: bool = True,
        use_quantized: bool = True,
        return_names: bool = False
    ):
        self.train = train
        self.use_quantized = use_quantized
        self.size = size
        self.return_names = return_names
        self.data_dir = data_dir
        self.animal_type = animal_type

        if train:
            path = os.path.join(data_dir, 'afhq', 'train', animal_type)
        else:
            path = os.path.join(data_dir, 'afhq', 'test', animal_type)

        if use_quantized:
            self.dataset = [os.path.join(path, image) for image in os.listdir(path) if image.endswith('.npy')]
        else:
            self.dataset = [os.path.join(path, image) for image in os.listdir(path) if image.endswith('.png')]

    def __getitem__(self, index):
        if self.use_quantized:
            image = torch.from_numpy(np.load(self.dataset[index]))
        else:
            transform = transforms.Compose([
                transforms.Resize((self.size, self.size)),
                transforms.ToTensor(),
            ])
            image = Image.open(self.dataset[index])
            image = image.convert('RGB')
            image = transform(image)

        if self.return_names:
           return image, self.dataset[index].split('/')[-1]
        return image

    def __len__(self):
        return len(self.dataset)
    
    def get_by_filename(self, index):
        transform = transforms.Compose([
                transforms.Resize((self.size, self.size)),
                transforms.ToTensor(),
        ])
        image = Image.open(os.path.join(self.data_dir, 'afhq', self.animal_type, index))
        image = image.convert('RGB')    
        image = transform(image)
        return image
    
    def repeat(self, n: int, max_len: int):
        self.dataset = self.dataset * n
        self.dataset = self.dataset[:max_len]

    @staticmethod
    def quantize_train(
        model: nn.Module, 
        data_dir: str,
        size: int = 128, 
        batch_size: int = 32,
    ):
        load_transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
        ])

        data_dir = os.path.join(data_dir, 'afhq')
        dataset = ImageFolder(data_dir, transform=load_transform, allow_empty=True) # allow_eppty because it will be handled in next line

        dataloader = DataLoader(dataset, batch_size=batch_size)
        for images, image_paths in tqdm(dataloader, file=sys.stdout):
            images = images.to(model.device)
            encoded_images = model.encode_to_cats(images).cpu().detach().numpy()
            for encoded_image, image_path in zip(encoded_images, image_paths):
                image_path = image_path.split('/')
                file_path = image_path[:-1]
                file_name = image_path[-1].split('.')[0]
                image_path = os.path.join(*file_path, file_name)
                np.save(image_path, encoded_image)


class YelpDataset(BaseDataset):
    def __init__(
        self, 
        sentiment: Literal['positive', 'negative', 'all'],
        data_dir: str, 
        tokenizer: Optional[PreTrainedTokenizerFast] = None,
        max_length: Optional[int] = None,
        split: Literal['train', 'eval', 'test', 'all', 'with_reference'] = 'train',
    ):
        self.sentiment = sentiment
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.split = split
        if tokenizer is not None:
            assert max_length is not None, 'max_length should be set if tokenizer is set!'

        self.file_path = os.path.join(data_dir, 'yelp', f'yelp_small_{split}.jsonl')
        
        self.file_positions = []
        with open(self.file_path, 'r') as f:
            pos, line = f.tell(), f.readline()
            while line:
                data = json.loads(line)
                stars = data['stars']
                if sentiment == 'positive' and stars >= 4:
                    self.file_positions.append(pos)
                elif sentiment == 'negative' and stars <= 2:
                    self.file_positions.append(pos)
                elif sentiment == 'all':
                    self.file_positions.append(pos)
                pos, line = f.tell(), f.readline()
    
    def __len__(self):
        return len(self.file_positions)

    def __getitem__(self, idx):
        file_pos = self.file_positions[idx]
        with open(self.file_path, 'r') as f:
            f.seek(file_pos)
            line = f.readline()
            data = json.loads(line)

        text = data['text']
        if self.tokenizer is not None:
            text = self.tokenizer.encode(
                text=text, 
                padding='max_length', 
                truncation=True, 
                max_length=self.max_length,
                return_tensors='pt',
                return_token_type_ids=False,
                return_attention_mask=False,
            ).squeeze() # type: ignore

            if self.split == 'with_reference':
                reference = data['reference']
                reference = self.tokenizer.encode(
                    text=reference, 
                    padding='max_length', 
                    truncation=True, 
                    max_length=self.max_length,
                    return_tensors='pt',
                    return_token_type_ids=False,
                    return_attention_mask=False,
                ).squeeze() # type: ignore
                return text, reference
            
        return text

    def repeat(self, n: int, max_len: int):
        original_positions = self.file_positions.copy()
        self.file_positions = []
        for _ in range(n):
            self.file_positions.extend(original_positions)
            if len(self.file_positions) >= max_len:
                self.file_positions = self.file_positions[:max_len]
                break


class AmazonDataset(BaseDataset):
    def __init__(
        self, 
        sentiment: Literal['positive', 'negative', 'all'],
        data_dir: str, 
        tokenizer: Optional[PreTrainedTokenizerFast] = None,
        max_length: Optional[int] = None,
        split: Literal['train', 'eval', 'test', 'all'] = 'train',
    ):
        self.sentiment = sentiment
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.split = split
        if tokenizer is not None:
            assert max_length is not None, 'max_length should be set if tokenizer is set!'

        self.file_path = os.path.join(data_dir, 'amazon', f'amazon_small_{split}.jsonl')
        
        self.file_positions = []
        with open(self.file_path, 'r') as f:
            pos, line = f.tell(), f.readline()
            while line:
                data = json.loads(line)
                if sentiment == 'positive' and data['sentiment'] == 'positive':
                    self.file_positions.append(pos)
                elif sentiment == 'negative' and data['sentiment'] == 'negative':
                    self.file_positions.append(pos)
                elif sentiment == 'all':
                    self.file_positions.append(pos)
                pos, line = f.tell(), f.readline()
    
    def __len__(self):
        return len(self.file_positions)

    def __getitem__(self, idx):
        file_pos = self.file_positions[idx]
        with open(self.file_path, 'r') as f:
            f.seek(file_pos)
            line = f.readline()
            data = json.loads(line)

        text = data['text']
        if self.tokenizer is not None:
            text = self.tokenizer.encode(
                text=text, 
                padding='max_length', 
                truncation=True, 
                max_length=self.max_length,
                return_tensors='pt',
                return_token_type_ids=False,
                return_attention_mask=False,
            ).squeeze() # type: ignore
            
        return text

    def repeat(self, n: int, max_len: int):
        original_positions = self.file_positions.copy()
        self.file_positions = []
        for _ in range(n):
            self.file_positions.extend(original_positions)
            if len(self.file_positions) >= max_len:
                self.file_positions = self.file_positions[:max_len]
                break


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
def get_cum_matrices(num_timesteps: int, onestep_matrix: torch.Tensor) -> torch.Tensor:
    num_categories = onestep_matrix.shape[0]
    cum_matrices = torch.empty(size=(num_timesteps, num_categories, num_categories), dtype=onestep_matrix.dtype)
    cum_matrices[0] = torch.eye(num_categories, dtype=onestep_matrix.dtype)
    
    for timestep in range(1, num_timesteps):
        cum_matrices[timestep] = cum_matrices[timestep-1] @ onestep_matrix
    
    assert onestep_matrix.shape == cum_matrices[0].shape, f'Wrong shape!'
    return cum_matrices


def uniform_prior(
    alpha: float, 
    num_categories: int, 
    num_timesteps: int,
    num_skip_steps: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    p_onestep_mat = torch.tensor([alpha / (num_categories - 1)] * num_categories**2, dtype=torch.float32)
    p_onestep_mat = p_onestep_mat.view(num_categories, num_categories)
    p_onestep_mat -= torch.diag(torch.diag(p_onestep_mat))
    p_onestep_mat += torch.diag(torch.tensor([1 - alpha] * num_categories))
    p_onestep_mat = torch.matrix_power(p_onestep_mat, n=num_skip_steps)

    p_cum_mats = get_cum_matrices(num_timesteps + 2, p_onestep_mat)

    return p_onestep_mat.transpose(0, 1), p_cum_mats


def von_mises_prior(
    alpha: float,
    num_categories: int, 
    num_timesteps: int,
    num_skip_steps: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    angles = np.linspace(0, 2 * np.pi, num_categories, endpoint=False)
    p_onestep_mat = np.zeros((num_categories, num_categories))

    for i, current_angle in enumerate(angles):
        for j, next_angle in enumerate(angles):
            p_onestep_mat[i, j] = np.exp((1 / (alpha**2 * (num_categories - 1)**2)) * np.cos(next_angle - current_angle))
        p_onestep_mat[i] /= np.sum(p_onestep_mat[i])
    p_onestep_mat = np.linalg.matrix_power(p_onestep_mat, n=num_skip_steps)

    p_onestep_mat = torch.from_numpy(p_onestep_mat)
    p_cum_mats = get_cum_matrices(num_timesteps + 2, p_onestep_mat)

    return p_onestep_mat.transpose(0, 1), p_cum_mats


def gaussian_prior(
    alpha: float,
    num_categories: int, 
    num_timesteps: int, 
    num_skip_steps: int,
    use_doubly_stochastic: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    p_onestep_mat = np.zeros([num_categories, num_categories], dtype=np.float32)
    max_distance = num_categories - 1
    if not use_doubly_stochastic:
        indices = np.arange(num_categories)[None, ...]
        values = (-4 * (indices - indices.T)**2) / ((alpha * max_distance)**2)
        p_onestep_mat = softmax(values, axis=1)
    else: # this logic mathcing D3PM article
        norm_const = -4 * (np.arange(-max_distance, max_distance+2, step=1, dtype=np.float32) ** 2)
        norm_const /= (alpha * max_distance)**2
        norm_const = np.exp(norm_const).sum()
        for i in range(num_categories):
            for j in range(num_categories):
                if i == j:
                    continue
                value = np.exp(-(4 * (i - j)**2) / (alpha * max_distance)**2)
                p_onestep_mat[i][j] = value / norm_const
        for i in range(num_categories):
            p_onestep_mat[i][i] = 1 - p_onestep_mat[i].sum() 

    p_onestep_mat = np.linalg.matrix_power(p_onestep_mat, n=num_skip_steps)
    p_onestep_mat = torch.from_numpy(p_onestep_mat) # .softmax(dim=1)
    p_cum_mats = get_cum_matrices(num_timesteps + 2, p_onestep_mat)

    return p_onestep_mat.transpose(0, 1), p_cum_mats


def centroid_gaussian_prior(
    alpha: float,
    num_categories: int,
    num_timesteps: int, 
    num_skip_steps: int,
    centroids: torch.Tensor | np.ndarray # num_categories x seq_length
) -> Tuple[torch.Tensor, torch.Tensor]:
    centroids = convert_to_numpy(centroids)
    distances = cdist(centroids, centroids, metric='euclidean')  # num_categories x num_categories
    max_distance = distances.max()

    p_onestep_mat = softmax(-distances / (alpha * max_distance)**2, axis=1)
    p_onestep_mat = np.linalg.matrix_power(p_onestep_mat, n=num_skip_steps)

    p_onestep_mat = torch.from_numpy(p_onestep_mat) 
    p_cum_mats = get_cum_matrices(num_timesteps + 2, p_onestep_mat)

    return p_onestep_mat.transpose(0, 1), p_cum_mats


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
        alpha: float,
        num_categories: int,
        num_timesteps: int,
        num_skip_steps: int,
        prior_type: Literal[
            'uniform', 
            'gaussian',
            'centroid_gaussian',
            'von_mises',
        ] = 'uniform',
        centroids: Optional[torch.Tensor] = None,
        eps: float = 1e-20,
        dtype: torch.dtype = torch.float32
    ) -> None:
        super().__init__()
        self.num_categories = num_categories
        self.num_timesteps = num_timesteps
        self.num_skip_steps = num_skip_steps
        self.eps = eps
        self.prior_type = prior_type
        self.dtype = dtype

        if prior_type == 'gaussian':
            p_onestep, p_cum = gaussian_prior(alpha, num_categories, num_timesteps, num_skip_steps)
        elif prior_type == 'von_mises':
            p_onestep, p_cum = von_mises_prior(alpha, num_categories, num_timesteps, num_skip_steps)
        elif prior_type == 'uniform':
            p_onestep, p_cum = uniform_prior(alpha, num_categories, num_timesteps, num_skip_steps)
        elif prior_type == 'centroid_gaussian' and centroids is not None:
            p_onestep, p_cum = centroid_gaussian_prior(alpha, num_categories, num_timesteps, num_skip_steps, centroids)
        else:
            raise NotImplementedError(f'Got unknown prior: {prior_type} or centroids is None!')
        self.register_buffer("p_onestep", p_onestep.to(dtype=dtype))
        self.register_buffer("p_cum", p_cum.to(dtype=dtype))
        
    def extract(
        self, 
        mat_type: Literal['onestep', 'cumulative'], 
        t: torch.Tensor, 
        *,
        row_id: Optional[torch.Tensor] = None,
        column_id: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Extracts row/column/element from transition matrix."""     
        if row_id is not None and column_id is not None:
            t = broadcast(t, row_id.dim() - 1)
            if mat_type  == 'onestep':
                return self.p_onestep[row_id, column_id]
            else: 
                return self.p_cum[t, row_id, column_id]
            
        elif row_id is not None and column_id is None:
            t = broadcast(t, row_id.dim() - 1)
            if mat_type  == 'onestep':
                return self.p_onestep[row_id]
            else: 
                return self.p_cum[t, row_id, :]
        
        elif row_id is None and column_id is not None:
            t = broadcast(t, column_id.dim() - 1)
            if mat_type  == 'onestep':
                return self.p_onestep[:, column_id]
            else:
                return self.p_cum[t, :, column_id]
        else:   
            raise ValueError('row_id and column_id cannot be None both!')

    def sample_bridge(self, x_start: torch.Tensor, x_end: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        r"""Samples from bridge $p(x_{t} | x_{0}, x_{1})$."""
        p_start_t = self.extract('cumulative', t, row_id=x_start)
        p_t_end = self.extract('cumulative', self.num_timesteps + 1 - t, column_id=x_end)
        log_probs = torch.log(p_start_t + self.eps) + torch.log(p_t_end + self.eps)
        log_probs = log_probs - log_probs.logsumexp(dim=-1, keepdim=True)
        
        noise = torch.rand_like(log_probs)
        noise = torch.clamp(noise, min=torch.finfo(noise.dtype).tiny, max=1.)
        gumbel_noise = -torch.log(-torch.log(noise))
        x_t = torch.argmax(log_probs + gumbel_noise, dim=-1)

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
        x_start_logits = x_start_logits.to(self.dtype)
        # fact1 is "guess of x_{t}" from x_{t-1}
        fact1 = self.extract('onestep', t, row_id=x_t)

        # fact2 is "guess of x_{t-1}" from x_{0}
        x_start_probs = x_start_logits.softmax(dim=-1)  # bs, ..., num_categories
        fact2 = torch.einsum("b...c,bcd->b...d", x_start_probs, self.p_cum[t - 1])
        p_posterior_logits = torch.log(fact1 + self.eps) + torch.log(fact2 + self.eps)
        p_posterior_logits = p_posterior_logits - p_posterior_logits.logsumexp(dim=-1, keepdim=True) # Normalize
        
        # Use `torch.where` because when `t == 1` x_start_logits are actually x_0 already
        is_first_step = broadcast(t, x_t.dim()) == 1
        p_posterior_logits = torch.where(is_first_step, x_start_logits, p_posterior_logits)
        return p_posterior_logits

