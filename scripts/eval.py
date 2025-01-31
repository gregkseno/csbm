import argparse
import glob
import os
from typing import Literal, Optional
from PIL import Image
import pandas as pd
from tqdm.auto import tqdm

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from pytorch_fid.fid_score import compute_statistics_of_path, calculate_frechet_distance
from pytorch_fid.inception import InceptionV3
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

_CLIP_MODEL_NAME = "openai/clip-vit-large-patch14-336"
_CUDA_AVAILABLE = torch.cuda.is_available()
# The bandwidth parameter for the Gaussian RBF kernel. See the paper for more
# details.
_SIGMA = 10
# The following is used to make the metric more human readable. See the paper
# for more details.
_SCALE = 1000


def _resize_bicubic(images, size):
    images = torch.from_numpy(images.transpose(0, 3, 1, 2))
    images = torch.nn.functional.interpolate(images, size=(size, size), mode="bicubic")
    images = images.permute(0, 2, 3, 1).numpy()
    return images

def mmd(x, y):
    """Memory-efficient MMD implementation in JAX.

    This implements the minimum-variance/biased version of the estimator described
    in Eq.(5) of
    https://jmlr.csail.mit.edu/papers/volume13/gretton12a/gretton12a.pdf.
    As described in Lemma 6's proof in that paper, the unbiased estimate and the
    minimum-variance estimate for MMD are almost identical.

    Note that the first invocation of this function will be considerably slow due
    to JAX JIT compilation.

    Args:
      x: The first set of embeddings of shape (n, embedding_dim).
      y: The second set of embeddings of shape (n, embedding_dim).

    Returns:
      The MMD distance between x and y embedding sets.
    """
    x = torch.from_numpy(x)
    y = torch.from_numpy(y)

    x_sqnorms = torch.diag(torch.matmul(x, x.T))
    y_sqnorms = torch.diag(torch.matmul(y, y.T))

    gamma = 1 / (2 * _SIGMA**2)
    k_xx = torch.mean(
        torch.exp(-gamma * (-2 * torch.matmul(x, x.T) + torch.unsqueeze(x_sqnorms, 1) + torch.unsqueeze(x_sqnorms, 0)))
    )
    k_xy = torch.mean(
        torch.exp(-gamma * (-2 * torch.matmul(x, y.T) + torch.unsqueeze(x_sqnorms, 1) + torch.unsqueeze(y_sqnorms, 0)))
    )
    k_yy = torch.mean(
        torch.exp(-gamma * (-2 * torch.matmul(y, y.T) + torch.unsqueeze(y_sqnorms, 1) + torch.unsqueeze(y_sqnorms, 0)))
    )

    return _SCALE * (k_xx + k_yy - 2 * k_xy)

class ClipEmbeddingModel:
    """CLIP image embedding calculator."""

    def __init__(self):
        self.image_processor = CLIPImageProcessor.from_pretrained(_CLIP_MODEL_NAME)

        self._model = CLIPVisionModelWithProjection.from_pretrained(_CLIP_MODEL_NAME).eval()
        if _CUDA_AVAILABLE:
            self._model = self._model.cuda() # type: ignore

        self.input_image_size = self.image_processor.crop_size["height"]

    @torch.no_grad()
    def embed(self, images):
        """Computes CLIP embeddings for the given images.

        Args:
          images: An image array of shape (batch_size, height, width, 3). Values are
            in range [0, 1].

        Returns:
          Embedding array of shape (batch_size, embedding_width).
        """
        images = _resize_bicubic(images, self.input_image_size)
        inputs = self.image_processor(
            images=images,
            do_normalize=True,
            do_center_crop=False,
            do_resize=False,
            do_rescale=False,
            return_tensors="pt",
        )
        if _CUDA_AVAILABLE:
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        image_embs = self._model(**inputs).image_embeds.cpu()
        image_embs /= torch.linalg.norm(image_embs, axis=-1, keepdims=True)
        return image_embs

class CMMDDataset(Dataset):
    def __init__(self, path, reshape_to, max_count=-1):
        self.path = path

        self.max_count = max_count
        img_path_list = self._get_image_list()
        if max_count > 0:
            img_path_list = img_path_list[:max_count]
        self.img_path_list = img_path_list

    def __len__(self):
        return len(self.img_path_list)

    def _get_image_list(self):
        ext_list = ["png", "jpg", "jpeg"]
        image_list = []
        for ext in ext_list:
            image_list.extend(glob.glob(f"{self.path}/*{ext}"))
            image_list.extend(glob.glob(f"{self.path}/*.{ext.upper()}"))
        # Sort the list to ensure a deterministic output.
        image_list.sort()
        return image_list

    def _read_image(self, path):
        im = Image.open(path)
        return np.asarray(im).astype(np.float32)

    def __getitem__(self, idx):
        img_path = self.img_path_list[idx]
        x = self._read_image(img_path)
        return x

def compute_embeddings_for_dir(
    img_dir,
    embedding_model,
    batch_size,
    max_count=-1,
):
    """Computes embeddings for the images in the given directory.

    This drops the remainder of the images after batching with the provided
    batch_size to enable efficient computation on TPUs. This usually does not
    affect results assuming we have a large number of images in the directory.

    Args:
      img_dir: Directory containing .jpg or .png image files.
      embedding_model: The embedding model to use.
      batch_size: Batch size for the embedding model inference.
      max_count: Max number of images in the directory to use.

    Returns:
      Computed embeddings of shape (num_images, embedding_dim).
    """
    dataset = CMMDDataset(img_dir, reshape_to=embedding_model.input_image_size, max_count=max_count)
    count = len(dataset)
    print(f"Calculating embeddings for {count} images from {img_dir}.")

    dataloader = DataLoader(dataset, batch_size=batch_size)

    all_embs = []
    for batch in tqdm(dataloader, total=count // batch_size):
        image_batch = batch.numpy()

        # Normalize to the [0, 1] range.
        image_batch = image_batch / 255.0

        if np.min(image_batch) < 0 or np.max(image_batch) > 1:
            raise ValueError(
                "Image values are expected to be in [0, 1]. Found:" f" [{np.min(image_batch)}, {np.max(image_batch)}]."
            )

        # Compute the embeddings using a pmapped function.
        embs = np.asarray(
            embedding_model.embed(image_batch)
        )  # The output has shape (num_devices, batch_size, embedding_dim).
        all_embs.append(embs)

    all_embs = np.concatenate(all_embs, axis=0)

    return all_embs

class PairedCelebaDataset(Dataset):  
    def __init__(
        self, 
        ref_sex: Literal['male', 'female'], 
        data_dir: str,
        eval_dir: str,
        size: int = 128, 
        max_count: int = -1
    ):
        self.size = size
        self.eval_dir = eval_dir
        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
        ])

        attrs = pd.read_csv(os.path.join(data_dir, 'list_attr_celeba.csv'))
        if ref_sex == 'male':
            attrs = attrs[attrs['Male'] != -1] # only males
        else:
            attrs = attrs[attrs['Male'] == -1]
        split = pd.read_csv(os.path.join(data_dir, 'list_eval_partition.csv'))

        split = split[split['partition'] != 0]
        image_names = pd.merge(attrs, split, on=['image_id'], how='inner')
        image_names = image_names['image_id'].tolist()
        self.reference = [os.path.join(data_dir, 'img_align_celeba', 'raw', image) for image in image_names][:max_count]
        self.generation = [os.path.join(eval_dir, image) for image in image_names][:max_count]

    def __getitem__(self, index):
        ref = Image.open(self.reference[index])
        ref = ref.convert('RGB')
        ref = self.transform(ref)

        gen = Image.open(self.generation[index])
        gen = gen.convert('RGB')
        gen = self.transform(gen)
        return ref, gen

    def __len__(self):
        return len(self.reference)

def calculate_cmmd(
    eval_dir: str, 
    embs_ref: np.ndarray, 
    batch_size: int = 32, 
    max_count: int = -1
):
    embedding_model = ClipEmbeddingModel()
    eval_embs = compute_embeddings_for_dir(eval_dir, embedding_model, batch_size, max_count).astype("float32")
    val = mmd(embs_ref, eval_embs)
    return val.numpy()

def calculate_fid(
    eval_dir: str,
    stats_ref: dict[str, np.ndarray], 
    dims: int = 2048,
    batch_size: int = 32,
    num_workers: int = 8,
    max_count: int = -1,
    device: str = 'cpu'
):
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx]).to(device)

    mu_ref, sigma_ref = stats_ref['mu'], stats_ref['sigma']
    mu_gen, sigma_gen = compute_statistics_of_path(
        eval_dir, model, batch_size, dims, device, num_workers, max_count
    )
    return calculate_frechet_distance(mu1=mu_gen, sigma1=sigma_gen, mu2=mu_ref, sigma2=sigma_ref)
    
def calculate_mse(
    eval_dir: str,
    ref_dir: str, 
    num_workers: int = 8,
    batch_size: int = 32, 
    max_count: int = -1
):
    dataset = PairedCelebaDataset(ref_sex='male', data_dir=ref_dir, eval_dir=eval_dir, max_count=max_count)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
    cost = 0
    for ref, gen in tqdm(dataloader):
        cost += (F.mse_loss(ref, gen) * batch_size).item()
    cost = cost / len(dataloader.dataset) # type: ignore
    return cost


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gen_data_path', type=str, required=True)
    parser.add_argument('--ref_data_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--dims', type=int, default=2048)
    parser.add_argument('--num_workers', type=int, default=8)
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    stats_ref_path = os.path.join(args.ref_data_path, 'fid_stats.npz')
    embd_ref_path = os.path.join(args.ref_data_path, 'cmmd_embed.npy')

    if os.path.exists(stats_ref_path):
        stats_ref = np.load(stats_ref_path).astype("float32")
    else:
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[args.dims]
        model = InceptionV3([block_idx]).to(device)
        m, s = compute_statistics_of_path(
            os.path.join(args.ref_data_path, 'img_align_celeba', 'raw'),
            model, args.batch_size,
            args.dims, device, args.num_workers, args.max_count
        )
        stats_ref = {'mu': m, 'sigma': s}
        np.savez(stats_ref_path, mu=m, sigma=s)
    
    if os.path.exists(embd_ref_path):
        embs_ref = np.load(embd_ref_path).astype("float32")
    else:
        embs_ref = compute_embeddings_for_dir(
            os.path.join(args.ref_data_path, 'img_align_celeba', 'raw'),
            ClipEmbeddingModel(), 
            args.batch_size, args.max_count
        ).astype("float32")
        np.save(embd_ref_path, embs_ref)

    fid = calculate_fid(
        eval_dir=args.gen_data_path,
        stats_ref=stats_ref,
        dims=args.dims,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_count=args.max_count,
        device=device
    )
    print(f'FID: {fid}')

    cmmd = calculate_cmmd(
        eval_dir=args.gen_data_path,
        embs_ref=embs_ref,
        batch_size=args.batch_size,
        max_count=args.max_count,
    )
    print(f'CMMD: {cmmd}')

    mse = calculate_mse(
        eval_dir=args.gen_data_path,
        ref_dir=args.ref_data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_count=args.max_count,
    )
    print(f'MSE: {mse}')
