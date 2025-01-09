from argparse import Namespace
import importlib
import os
from datetime import datetime
from typing import Any, Dict, List, Literal, Mapping, Optional, Tuple
import json

from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from PIL import Image
import wandb

from accelerate.tracking import GeneralTracker
import numpy as np
import torch
from torchvision.utils import make_grid


class DotDict(Namespace):
    """A simple class that builds upon `argparse.Namespace`
    in order to make chained attributes possible.
    https://github.com/nicholasmireles/DotDict/tree/main
    """

    def __init__(self, temp=False, key=None, parent=None) -> None:
        self._temp = temp
        self._key = key
        self._parent = parent

    def __eq__(self, other):
        if not isinstance(other, DotDict):
            return NotImplemented
        return vars(self) == vars(other)

    def __getattr__(self, __name: str) -> Any:
        if __name not in self.__dict__ and not self._temp:
            self.__dict__[__name] = DotDict(temp=True, key=__name, parent=self)
        else:
            del self._parent.__dict__[self._key] # type: ignore
            raise AttributeError("No attribute '%s'" % __name)
        return self.__dict__[__name]

    def __repr__(self) -> str:
        return json.dumps(self.to_dict(), indent=4)

    @classmethod
    def from_dict(cls, original: Mapping[str, Any]) -> "DotDict":
        """Create a DotDict from a (possibly nested) dict `original`.
        Warning: this method should not be used on very deeply nested inputs,
        since it's recursively traversing the nested dictionary values.
        """
        dd = DotDict()
        for key, value in original.items():
            if isinstance(value, Mapping):
                value = cls.from_dict(value)
            setattr(dd, key, value)
        return dd
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert a DotDict back to a standard dictionary."""
        result = {}
        for key, value in self.__dict__.items():
            if key.startswith("_"):
                continue
            if isinstance(value, DotDict):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result

def create_expertiment(exp_dir: str, hyperparams: DotDict) -> Tuple[str, str]:
    """Creates directory for experiment.
    Returns experiment name and path."""
    dim_alpha = f'dim_{hyperparams.data.dim}_aplha_{hyperparams.prior.alpha}'
    if hyperparams.ce_loss_coeff == 0:
        dim_alpha += '_no_ce'
    if hyperparams.kl_loss_coeff == 0:
        dim_alpha += '_no_kl'

    save_dir_name = os.path.join(exp_dir, hyperparams.data.type)
    if not os.path.exists(save_dir_name):
        os.mkdir(save_dir_name)

    prior = f'{hyperparams.prior.type}'
    save_dir_name = os.path.join(save_dir_name, prior)
    if not os.path.exists(save_dir_name):
        os.mkdir(save_dir_name)

    time = datetime.now().strftime("%d.%m.%y_%H:%M:%S")
    save_dir_name = os.path.join(save_dir_name, dim_alpha + '_' + time)
    if not os.path.exists(save_dir_name):
        os.mkdir(save_dir_name)

    return '_'.join([hyperparams.data.type, prior, dim_alpha]), save_dir_name

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

def visualize(
    exp_type: Literal['toy', 'images', 'quantized_images'],
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
        fig_path = os.path.join(exp_path, f'samples_{fb}_{iteration}.png')
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
        fig_path = os.path.join(exp_path, f'samples_{fb}_{iteration}.png')
        if not os.path.isfile(fig_path):
            im.save(fig_path)
    
    if tracker:
        tracker.log({f'samples_{fb}': [wandb.Image(im)]}, step=step)

    plt.close()


def visualize_trajectory(
    exp_type: Literal['toy', 'images', 'quantized_images'],
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
    
    pred_x_start = convert_to_numpy(pred_x_start)
    trajectories = convert_to_numpy(trajectories)
    num_trajectories = trajectories.shape[1]

    ax.scatter(
        pred_x_start[:, 0], pred_x_start[:, 1], 
        c="salmon", s=64, edgecolors="black", 
        label = "Fitted distribution", zorder=1
    )
    ax.scatter(
        trajectories[0, :, 0], trajectories[0, :, 1], 
        c="lime", s=128, edgecolors="black", 
        label=r"Trajectory start ($x \sim p_0$)", zorder=3
    )
    ax.scatter(
        trajectories[-1, :, 0], trajectories[-1, :, 1], 
        c="yellow", s=64, edgecolors="black", 
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
    
    ax.legend(loc="lower right")
    plt.show()
    
    fig.tight_layout(pad=0.5)
    im = fig2img(fig)

    if exp_path is not None:
        fig_path = os.path.join(exp_path, f'trajectories_{fb}_{iteration}.png')
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
    num_timesteps = trajectories.shape[0]
    trajectories = torch.stack([
            trajectories[0], 
            trajectories[num_timesteps // 8], 
            trajectories[num_timesteps // 2], 
            trajectories[(num_timesteps * 7) // 8], 
            trajectories[-1]
        ], dim=0
    )
    num_timesteps, batch_size = trajectories.shape[:2]
    trajectories = trajectories.reshape(num_timesteps * batch_size, *trajectories.shape[2:])
    trajectories = convert_to_numpy(make_grid(trajectories, nrow=batch_size))

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
        fig_path = os.path.join(exp_path, f'trajectories_{fb}_{iteration}.png')
        if not os.path.isfile(fig_path):
            im.save(fig_path)
    
    if tracker:
        tracker.log({f'trajectories_{fb}': [wandb.Image(im)]}, step=step)

    plt.close()
