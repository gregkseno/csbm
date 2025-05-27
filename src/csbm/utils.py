import os
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig

from datetime import datetime
from typing import Any, List, Literal, Optional, Tuple, Callable

import json
import logging
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from PIL import Image
import wandb

from accelerate.tracking import GeneralTracker, on_main_process
from accelerate.logging import get_logger
import numpy as np
import torch
from torchvision.utils import make_grid


def create_expertiment(exp_dir: str, hyperparams: DictConfig | ListConfig) -> Tuple[str, str]:
    """Creates directory for experiment.
    Returns experiment name and path."""
    dim_alpha = f'dim_{hyperparams.data.dim}_aplha_{hyperparams.prior.alpha}'
    if hyperparams.train.ce_loss_coeff == 0.:
        dim_alpha += '_no_ce'
    if hyperparams.train.kl_loss_coeff == 0.:
        dim_alpha += '_no_kl'
    prior = f'{hyperparams.prior.type}'
    time = datetime.now().strftime("%d.%m.%y_%H:%M:%S")
    
    save_dir_name = os.path.join(
        exp_dir, hyperparams.data.type, hyperparams.data.dataset, prior, dim_alpha + '_' + time
    )
    os.makedirs(os.path.join(save_dir_name, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(save_dir_name, 'samples'), exist_ok=True)
    os.makedirs(os.path.join(save_dir_name, 'trajectories'), exist_ok=True)
    return '_'.join([hyperparams.data.type, hyperparams.data.dataset, prior, dim_alpha]), save_dir_name

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

def broadcast(t: torch.Tensor, num_add_dims: int) -> torch.Tensor:
    shape = [t.shape[0]] + [1] * num_add_dims
    return t.reshape(shape)

class ConsoleTracker(GeneralTracker):
    name = "console"
    requires_logging_directory = False

    def __init__(self, print_fn: Optional[Callable] = None):
        if print_fn is None:
            self.print_fn = print

    @property
    def tracker(self):
        return self.print_fn

    @on_main_process
    def store_init_configuration(self, values: dict):
        self.print_fn("\n".join([f"{key}: {value}" for key, value in values.items()]))

    @on_main_process
    def log(self, values: dict, step: Optional[int] = None):
        self.print_fn(" ".join([f"{key}: {value}" for key, value in values.items()]))

def visualize(
    exp_type: Literal['toy', 'images', 'quantized_images', 'texts'],
    x_end: Any, 
    x_start: Any, 
    pred_x_start: Any, 
    fb: Literal['forward', 'backward'],
    labels: Optional[List[str]] = None,
    iteration: Optional[int] = None, 
    exp_path: Optional[str] = None, 
    tracker: Optional[GeneralTracker] = None,
    step: Optional[int] = None,
):
    if exp_type == 'toy':
        vis_func = visualize_toy
    elif exp_type == 'images' or exp_type == 'quantized_images':
        vis_func = visualize_images
    elif exp_type == 'texts':
        vis_func = visualize_texts
    else:
        raise NotImplementedError(f"Unknown exp type {exp_type}!")
    vis_func(
        x_end=x_end, 
        x_start=x_start, 
        pred_x_start=pred_x_start, 
        tracker=tracker,
        fb=fb, 
        labels=labels, 
        iteration=iteration, 
        exp_path=exp_path, 
        step=step
    )

def visualize_toy(
    x_end: torch.Tensor | np.ndarray, 
    x_start: torch.Tensor | np.ndarray, 
    pred_x_start: torch.Tensor | np.ndarray, 
    fb: Literal['forward', 'backward'],
    labels: Optional[List[str]] = None,
    iteration: Optional[int] = None, 
    exp_path: Optional[str] = None, 
    tracker: Optional[GeneralTracker] = None,
    step: Optional[int] = None
):
    x_start = convert_to_numpy(x_start)
    x_end = convert_to_numpy(x_end)
    pred_x_start = convert_to_numpy(pred_x_start)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4),squeeze=True,sharex=True,sharey=True)
    if iteration is not None:
        fig.suptitle(f'Iteration {iteration}')

    axs_max = np.abs(np.concatenate([x_start, x_end, pred_x_start], axis=0)).max() + 0.5

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
        fig_path = os.path.join(exp_path, 'samples', f'samples_{fb}_{iteration}_step_{step}.png')
        if not os.path.isfile(fig_path):
            im.save(fig_path)

    if tracker:
        tracker.log({f'samples_{fb}': [wandb.Image(im)]}, step=step)

    plt.close()

def visualize_images(
    x_end: torch.Tensor | np.ndarray, 
    x_start: torch.Tensor | np.ndarray, 
    pred_x_start: torch.Tensor | np.ndarray, 
    fb: Literal['forward', 'backward'],
    labels: Optional[List[str]] = [r'$p_0$', r'$p_1$', r'$p_{\theta}$'],
    iteration: Optional[int] = None, 
    exp_path: Optional[str] = None, 
    tracker: Optional[GeneralTracker] = None,
    step: Optional[int] = None
):
    nrow = int(x_start.shape[0]**0.5)
    x_start = convert_to_numpy(make_grid(convert_to_torch(x_start), nrow=nrow))
    x_end = convert_to_numpy(make_grid(convert_to_torch(x_end), nrow=nrow))
    pred_x_start = convert_to_numpy(make_grid(convert_to_torch(pred_x_start), nrow=nrow))

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), squeeze=True, sharex=True, sharey=True)
    if iteration is not None:
        fig.suptitle(f'Iteration {iteration}')

    axes[0].imshow(x_end.transpose(1, 2, 0)) 
    axes[1].imshow(x_start.transpose(1, 2, 0))
    axes[2].imshow(pred_x_start.transpose(1, 2, 0))
    
    for i in range(3):
        axes[i].get_xaxis().set_ticklabels([])
        axes[i].get_yaxis().set_ticklabels([])
        axes[i].set_axis_off()
        if labels is not None:
            axes[i].set_title(labels[i])
            
    plt.show()
    fig.tight_layout(pad=0.5)
    im = fig2img(fig)

    if exp_path is not None:
        fig_path = os.path.join(exp_path, 'samples', f'samples_{fb}_{iteration}_step_{step}.png')
        if not os.path.isfile(fig_path):
            im.save(fig_path)
    
    if tracker:
        tracker.log({f'samples_{fb}': [wandb.Image(im)]}, step=step)

    plt.close()

def visualize_texts(
    x_end: List[str], 
    x_start: List[str], 
    pred_x_start: List[str], 
    fb: Literal['forward', 'backward'],
    labels: Optional[List[str]] = [r'$p_0$', r'$p_1$', r'$p_{\theta}$'],
    iteration: Optional[int] = None, 
    exp_path: Optional[str] = None, 
    tracker: Optional[GeneralTracker] = None,
    step: Optional[int] = None
):
    sample_triplets = list(zip(x_end, pred_x_start, x_start))
    if exp_path is not None:
        jsonl_path = os.path.join(exp_path, 'samples', f'samples_{fb}_{iteration}_step_{step}.jsonl')
        if not os.path.isfile(jsonl_path):
            with open(jsonl_path, 'w') as f:
                for init, generated, example in sample_triplets:
                    sample = {
                        "initial": init,
                        "generated": generated,
                        "example": example,
                    }
                    f.write(json.dumps(sample) + '\n')
                    
    if tracker:
        table = wandb.Table(columns=["Initial text", "Generated text", "Example text"])
        for init, generated, example in sample_triplets:
            table.add_data(init, generated, example)
        tracker.log({f'{fb}_text_samples': table}, step=step)

def visualize_trajectory(
    exp_type: Literal['toy', 'images', 'quantized_images', 'texts'],
    pred_x_start: torch.Tensor | np.ndarray, 
    trajectories: torch.Tensor | np.ndarray, 
    fb: Literal['forward', 'backward'],
    figsize: Tuple[float, float] | None = None,
    iteration: Optional[int] = None, 
    exp_path: Optional[str] = None,
    tracker: Optional[GeneralTracker] = None,
    step: Optional[int] = None
):
    if exp_type == 'toy':
        vis_func = visualize_trajectory_toy
    elif exp_type == 'images' or exp_type == 'quantized_images':
        vis_func = visualize_trajectory_image
    else:
        raise NotImplementedError(f"Unknown exp type {exp_type}!")
    vis_func(
        pred_x_start=pred_x_start,
        trajectories=trajectories, 
        tracker=tracker,
        fb=fb, 
        figsize=figsize, 
        iteration=iteration, 
        exp_path=exp_path, 
        step=step
    )
    
def visualize_trajectory_toy(
    pred_x_start: torch.Tensor | np.ndarray, 
    trajectories: torch.Tensor | np.ndarray, 
    fb: Literal['forward', 'backward'],
    figsize: Tuple[float, float] | None = None,
    iteration: Optional[int] = None, 
    exp_path: Optional[str] = None,
    tracker: Optional[GeneralTracker] = None,
    step: Optional[int] = None,
    dpi: int = 200,
    use_legend: bool = True,
    axlim: Optional[Tuple[float, float]] = None
):
    if figsize is None:
        figsize = (8, 8)
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    ax.grid(zorder=-20)
    ax.get_xaxis().set_ticklabels([])
    ax.get_yaxis().set_ticklabels([])
    if iteration is not None:
        fig.suptitle(f'Iteration {iteration}')
    
    pred_x_start = convert_to_numpy(pred_x_start)
    trajectories = convert_to_numpy(trajectories)
    num_trajectories = trajectories.shape[1]

    ax.scatter(
        pred_x_start[:, 0], pred_x_start[:, 1], 
        c="salmon", s=25, edgecolors="black", 
        label = "Fitted distribution", zorder=1, linewidth=0.5
    )
    ax.scatter(
        trajectories[0, :, 0], trajectories[0, :, 1], 
        c="lime", s=60, edgecolors="black", 
        label=r"Trajectory start ($x \sim p_0$)", zorder=3
    )
    ax.scatter(
        trajectories[-1, :, 0], trajectories[-1, :, 1], 
        c="yellow", s=22, edgecolors="black", 
        label = r"Trajectory end (fitted)", zorder=3
    )
    for i in range(num_trajectories):
        ax.plot(
            trajectories[:, i, 0], trajectories[:, i, 1], 
            "black", markeredgecolor="black", linewidth=1.5, zorder=2
        )
        if i == 0:
            ax.plot(
                trajectories[:, i, 0], trajectories[:, i, 1], 
                "grey", markeredgecolor="black", linewidth=0.5, zorder=2, 
                label="Intermediate predictions"
            )
        else:
            ax.plot(
                trajectories[:, i, 0], trajectories[:, i, 1], 
                "grey", markeredgecolor="black", linewidth=0.5, zorder=2
            )
    if use_legend:
        ax.legend(loc="upper left")

    if axlim is not None:
        ax.set_xlim(axlim)
        ax.set_ylim(axlim)
    plt.show()
    
    fig.tight_layout(pad=0.5)
    im = fig2img(fig)

    if exp_path is not None:
        fig_path = os.path.join(exp_path, 'trajectories', f'trajectories_{fb}_{iteration}_step_{step}.png')
        if not os.path.isfile(fig_path):
            im.save(fig_path)
    if tracker:
        tracker.log({f'trajectories_{fb}': [wandb.Image(im)]}, step=step)

    plt.close()


def visualize_trajectory_image(
    pred_x_start: torch.Tensor | np.ndarray, 
    trajectories: torch.Tensor | np.ndarray, 
    fb: Literal['forward', 'backward'],
    figsize: Tuple[float, float] | None = None,
    iteration: Optional[int] = None, 
    exp_path: Optional[str] = None,
    tracker: Optional[GeneralTracker] = None,
    step: Optional[int] = None
):
    if figsize is None:
        figsize = (8, 8)
    fig, ax = plt.subplots(1, 1, figsize=figsize)
      
    # Sampling
    batch_size = trajectories.shape[1]
    trajectories = trajectories.reshape(-1, *trajectories.shape[2:])
    trajectories = convert_to_numpy(make_grid(convert_to_torch(trajectories), nrow=batch_size))

    ax.imshow(trajectories.transpose(1, 2, 0)) 
    if iteration is not None:
        ax.set_title(f'Iteration {iteration}')
    ax.get_xaxis().set_ticklabels([])
    ax.get_yaxis().set_ticklabels([])
    ax.set_axis_off()
    plt.show()
    
    fig.tight_layout(pad=0.5)
    im = fig2img(fig)

    if exp_path is not None:
        fig_path = os.path.join(exp_path, 'trajectories', f'trajectories_{fb}_{iteration}_step_{step}.png')
        if not os.path.isfile(fig_path):
            im.save(fig_path)
    
    if tracker:
        tracker.log({f'trajectories_{fb}': [wandb.Image(im)]}, step=step)

    plt.close()
