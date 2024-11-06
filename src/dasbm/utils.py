import os
from datetime import datetime
from typing import List, Literal, Optional, Tuple
from argparse import Namespace

from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from PIL import Image
import wandb

import numpy as np
import torch
from torch import nn

def create_expertiment(hyperparams: Namespace) -> Tuple[str, str]:
    """Creates directory for experiment.
    Returns experiment name and path."""
    dim_alpha = f'dim_{hyperparams.input_dim}_aplha_{hyperparams.alpha}'

    save_dir_name = os.path.join('../experiments', hyperparams.exp_type)
    if not os.path.exists(save_dir_name):
        os.mkdir(save_dir_name)

    prior = f'{hyperparams.prior}'
    save_dir_name = os.path.join(save_dir_name, prior)
    if not os.path.exists(save_dir_name):
        os.mkdir(save_dir_name)

    time = datetime.now().strftime("%d.%m.%y_%H:%M")
    save_dir_name = os.path.join(save_dir_name, dim_alpha + '_' + time)
    if not os.path.exists(save_dir_name):
        os.mkdir(save_dir_name)

    return '_'.join([hyperparams.exp_type, prior, dim_alpha]), save_dir_name

def fig2img(fig: Figure) -> Image.Image:
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8) # type: ignore
    buf.shape = (w, h, 4)
    buf = np.roll(buf, 3, axis = 2)
    return Image.frombytes("RGBA", (w, h), buf.tobytes())

def convert_to_numpy(x: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return x
    
def convert_to_torch(x: torch.Tensor | np.ndarray) -> torch.Tensor:
    if isinstance(x, np.ndarray):
        x = torch.tensor(x)
    return x

def visualize(
        x_end: torch.Tensor | np.ndarray, 
        x_start: torch.Tensor | np.ndarray, 
        pred_x_start: torch.Tensor | np.ndarray, 
        fb: Literal['forward', 'backward'],
        labels: Optional[List[str]] = None,
        iteration: Optional[int] = None, 
        exp_path: Optional[str] = None, 
        step: Optional[int] = None
    ):
    x_start, x_end, pred_x_start = convert_to_numpy(x_start), convert_to_numpy(x_end), convert_to_numpy(pred_x_start)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4),squeeze=True,sharex=True,sharey=True)
    if iteration is not None:
        fig.suptitle(f'Iteration {iteration}')

    axs_max = np.abs(np.concat([x_start, x_end, pred_x_start], axis=0)).max() + 0.5

    label_x_start, label_x_end, label_pred_x_start = labels if labels is not None else (None, None, None)
    axes[0].scatter(x_end[:, 0], x_end[:, 1], c="orange", edgecolor='black', label=label_x_end, s=30) # type: ignore
    axes[1].scatter(x_start[:, 0], x_start[:, 1], c="g", edgecolor='black', label=label_x_start, s=30) # type: ignore
    axes[2].scatter(pred_x_start[:, 0], pred_x_start[:, 1], c="yellow", edgecolor='black', label=label_pred_x_start, s=30) # type: ignore
    
    for i in range(3):
        axes[i].grid() # type: ignore
        axes[i].set_xlim([0, axs_max]) # type: ignore
        axes[i].set_ylim([0, axs_max]) # type: ignore
        if labels is not None:
            axes[i].legend() # type: ignore
    plt.show()
    fig.tight_layout(pad=0.5)
    im = fig2img(fig)

    if exp_path is not None:
        fig_path = os.path.join(exp_path, f'samples_{fb}_{iteration}.png')
        im.save(fig_path)
        plt.savefig(fig_path)
    
    if wandb.run:
        wandb.log({f'samples_{fb}': [wandb.Image(im)]}, step=step)

def visualize_trajectory(
    x_end: torch.Tensor | np.ndarray, 
    model: nn.Module, 
    fb: Literal['forward', 'backward'],
    num_traslations: int = 2,
    figsize: Tuple[float, float] | None = None,
    iteration: Optional[int] = None, 
    exp_path: Optional[str] = None,
    step: Optional[int] = None
):
    if figsize is None:
        figsize = (8, 8)
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.grid(zorder=-20)
    ax.get_xaxis().set_ticklabels([])
    ax.get_yaxis().set_ticklabels([])
    if iteration is not None:
        fig.suptitle(f'Iteration {iteration}')
        
    tr_samples = torch.tensor([[0, 0], [7, 17], [15, 45], [20, 20]])
    tr_samples = (tr_samples
        .unsqueeze(0)
        .repeat(num_traslations, 1, 1)
        .reshape(num_traslations * 4, 2)
        .to(model.device)
    )
    
    # Sampling
    x_end = convert_to_torch(x_end)
    pred_x_start = model.sample(x_end).cpu()
    ax.scatter(pred_x_start[:, 0], pred_x_start[:, 1], c="salmon", s=64, edgecolors="black", label = "Fitted distribution", zorder=1)
    trajectory = convert_to_numpy(model.sample_trajectory(tr_samples))
    tr_samples = convert_to_numpy(tr_samples)
    
    ax.scatter(tr_samples[:, 0], tr_samples[:, 1], 
    c="lime", s=128, edgecolors="black", label = r"Trajectory start ($x \sim p_0$)", zorder=3)
    
    ax.scatter(trajectory[-1, :, 0], trajectory[-1, :, 1], 
    c="yellow", s=64, edgecolors="black", label = r"Trajectory end (fitted)", zorder=3)
    
    for i in range(num_traslations * 4):
        ax.plot(trajectory[::1, i, 0], trajectory[::1, i, 1], "black", markeredgecolor="black",
            linewidth=1.5, zorder=2)
        if i == 0:
            ax.plot(trajectory[::1, i, 0], trajectory[::1, i, 1], "grey", markeredgecolor="black",
                    linewidth=0.5, zorder=2, label="Intermediate predictions")
        else:
            ax.plot(trajectory[::1, i, 0], trajectory[::1, i, 1], "grey", markeredgecolor="black",
                    linewidth=0.5, zorder=2)
    
    ax.legend(loc="lower right")
    plt.show()
    
    fig.tight_layout(pad=0.5)
    im = fig2img(fig)

    if exp_path is not None:
        fig_path = os.path.join(exp_path, f'trajectories_{fb}_{iteration}.png')
        im.save(fig_path)
        plt.savefig(fig_path)
    
    if wandb.run:
        wandb.log({f'trajectories_{fb}': [wandb.Image(im)]}, step=step)

def broadcast(t: torch.Tensor, num_add_dims: int) -> torch.Tensor:
    shape = [t.shape[0]] + [1] * num_add_dims
    return t.reshape(shape)
