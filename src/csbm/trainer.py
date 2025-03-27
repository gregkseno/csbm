import os
import sys
from typing import Any, Literal, Optional, Union

from tqdm.auto import tqdm

from accelerate import Accelerator
import numpy as np
import ot
import torch
from torch.optim import Optimizer # type: ignore
from torch.utils.data import DataLoader, RandomSampler
from torch_ema import ExponentialMovingAverage as EMA
from torchmetrics.image.fid import FrechetInceptionDistance as FID

from csbm.models.quantized_images import LatentD3PM, Codec
from csbm.models.toy import D3PM
from csbm.models.images import ImageD3PM

from csbm.data import BaseDataset, CouplingDataset, Prior
from csbm.utils import visualize, visualize_trajectory
from csbm.vq_diffusion.engine.lr_scheduler import ReduceLROnPlateauWithWarmup


class СSBMTrainer:
    exp_type: Literal['toy', 'images', 'quantized_images']
    forward_and_backward = {'forward', 'backward'}
    checkpoint_path = 'EMPTY'

    def __init__(
        self,
        iterations: int,
        inner_iterations: int,
        prior_iterations: int,
        use_mini_batch: bool,
        accelerator: Accelerator,
        forward_model: Union[D3PM, ImageD3PM, LatentD3PM],
        backward_model: Union[D3PM, ImageD3PM, LatentD3PM],
        prior: Prior,
        forward_optimizer: Optimizer,
        backward_optimizer: Optimizer,
        forward_scheduler: Optional[ReduceLROnPlateauWithWarmup],
        backward_scheduler: Optional[ReduceLROnPlateauWithWarmup],
        codec: Optional[Codec] = None,
        exp_type: Literal['toy', 'images', 'quantized_images'] = 'toy', 
        exp_path: Optional[str] = None,
        kl_loss_coeff: float = 1.,
        ce_loss_coeff: float = 0.001,
        ema_decay: float = 0.999,
        eval_freq: int = 1000,
        num_trajectories: int = 4,
        num_translations: int = 5,
    ) -> None:
        assert kl_loss_coeff > 0 or ce_loss_coeff > 0, 'At least one loss coefficents must be greater than zero!'

        self.iterations = iterations
        self.inner_iterations = inner_iterations
        self.prior_iterations = prior_iterations
        self.use_mini_batch = use_mini_batch

        self.accelerator = accelerator
        self.models = {
            'forward': forward_model,
            'backward': backward_model
        }
        self.emas = {
            'forward': EMA(forward_model.parameters(), decay=ema_decay),
            'backward': EMA(backward_model.parameters(), decay=ema_decay)
        }
        self.prior = prior
        self.codec = codec
        self.optimizers = {
            'forward': forward_optimizer,
            'backward': backward_optimizer
        }
        self.schedulers = None
        if forward_scheduler is not None and backward_scheduler is not None:
            self.schedulers = {
                'forward': forward_scheduler,
                'backward': backward_scheduler
            }

        self.kl_loss_coeff = kl_loss_coeff
        self.ce_loss_coeff = ce_loss_coeff
        self.eps = 1e-6

        self.exp_type = exp_type
        self.exp_path = exp_path
        if exp_path is not None:
            self.checkpoint_path = os.path.join(exp_path, 'checkpoints')            

        self.is_generation_normalized = (exp_type == 'quantized_images')
        self.fids = {
            'forward': FID(
                feature=2048, 
                reset_real_features=False, 
                normalize=self.is_generation_normalized
            ).to(self.accelerator.device),
            'backward': FID(
                feature=2048, 
                reset_real_features=False, 
                normalize=self.is_generation_normalized
            ).to(self.accelerator.device)
        }
        self.eval_freq = eval_freq
        self.num_trajectories = num_trajectories
        self.num_translations = num_translations

    def kl_loss(
        self, 
        true_q_posterior_logits: Any, 
        pred_q_posterior_logits: Any,
    ) -> torch.Tensor:        
        """KL-divergence calculation."""    
        if self.exp_type == 'toy' or self.exp_type == 'images' or self.exp_type == 'quantized_images':
            kl_loss = torch.softmax(true_q_posterior_logits + self.eps, dim=-1) * (
                torch.log_softmax(true_q_posterior_logits + self.eps, dim=-1)
                - torch.log_softmax(pred_q_posterior_logits + self.eps, dim=-1)
            )
            kl_loss = kl_loss.sum(dim=-1).mean()
        else:
            raise NotImplementedError(f"Unknown exp type {self.exp_type}!")
        return kl_loss
    
    def ce_loss(
        self, 
        true_x_start: Any, 
        pred_x_start_logits: Any, 
    ) -> torch.Tensor:   
        """Cross-Entropy calculation."""         
        if self.exp_type == 'toy' or self.exp_type == 'images' or self.exp_type == 'quantized_images':
            pred_x_start_logits = pred_x_start_logits.flatten(start_dim=0, end_dim=-2)
            true_x_start = true_x_start.flatten(start_dim=0, end_dim=-1)
            ce_loss = torch.nn.CrossEntropyLoss()(pred_x_start_logits, true_x_start)
        else:
            raise NotImplementedError(f"Unknown exp type {self.exp_type}!")
        return ce_loss

    def markovian_projection(
        self,
        fb: Literal['forward', 'backward'],
        trainloader: DataLoader,
        testloader: DataLoader
    ):  
        bf = 'backward' if fb == 'forward' else 'forward'
        self.models[fb].train()
        self.models[bf].eval()

        trange = tqdm(
            trainloader, 
            desc=f'{fb.capitalize()} D-IMF iteration: {self.iteration}', 
            file=sys.stdout, 
            disable=not self.accelerator.is_local_main_process
        )
        for batch in trange:
            self.step += 1

            with self.accelerator.accumulate():
                self.optimizers[fb].zero_grad()

                if self.iteration == 1:
                    true_x_start, true_x_end = batch
                    if self.use_mini_batch:
                        pi = self._get_map(true_x_start.float(), true_x_end.float())
                        i, j = self._sample_map(pi, true_x_start.shape[0])
                        true_x_start, true_x_end = true_x_start[i], true_x_end[j]
                else:
                    true_x_start, true_x_end = batch, self.models[bf].sample(batch, self.prior)

                t = torch.randint(
                    low=1, 
                    high=self.models[fb].num_timesteps + 2,
                    size=(true_x_start.shape[0],), 
                    device=self.accelerator.device
                )
                x_t = self.prior.sample_bridge(true_x_start, true_x_end, t) # type: ignore

                loss = 0
                pred_x_start_logits = self.models[fb](x_t, t)

                # KL-loss calculation
                true_q_posterior_logits = self.prior.posterior_logits(true_x_start, x_t, t, logits=False) # Version of DDPM
                pred_q_posterior_logits = self.prior.posterior_logits(pred_x_start_logits, x_t, t, logits=True)
                kl_loss = self.kl_loss(true_q_posterior_logits, pred_q_posterior_logits)
                loss += self.kl_loss_coeff * kl_loss

                # CE-loss calculation
                ce_loss = self.ce_loss(true_x_start, pred_x_start_logits)
                loss += self.ce_loss_coeff * ce_loss

                self.accelerator.backward(loss)
                self.optimizers[fb].step()
                if self.schedulers is not None:
                    self.schedulers[fb].step(kl_loss.detach())
                self.emas[fb].update()

            info = {
                "kl_loss": self.accelerator.reduce(kl_loss.detach(), 'mean'),  # type: ignore
                "ce_loss": self.accelerator.reduce(ce_loss.detach(), 'mean')  # type: ignore
            }
                
            if self.step % self.eval_freq == 0:
                self.accelerator.print(f'{fb.capitalize()} D-IMF iteration: {self.iteration}: kl_loss: {info["kl_loss"]}, ce_loss: {info["ce_loss"]}')
                self.viz(fb=fb, dataloader=testloader, step=self.step)
            if self.step % (self.eval_freq * 5) == 0 and self.exp_type != 'toy':
                self.eval(fb=fb, dataloader=testloader, step=self.step)
            self.accelerator.log(info, step=self.step)
        
    def viz(
        self,
        fb: Literal['forward', 'backward'],
        dataloader: DataLoader, 
        step: Optional[int]
    ):
        self.models[fb].eval()
        with self.emas[fb].average_parameters():
            test_x_start, test_x_end = next(iter(dataloader))

            if self.codec is not None:
                encoded_test_x_end = self.codec.encode_to_cats(test_x_end)
                pred_x_start = self.models[fb].sample(encoded_test_x_end, self.prior)
                pred_x_start = self.codec.decode_to_image(pred_x_start)
            else:
                pred_x_start = self.models[fb].sample(test_x_end, self.prior)
                
            if self.codec is not None:
                traj_start = encoded_test_x_end[:self.num_trajectories]
            else:
                traj_start = test_x_end[:self.num_trajectories]
            repeats = [self.num_translations] + [1] * traj_start.dim()
            trajectories = traj_start.unsqueeze(0).repeat(*repeats)
            trajectories = trajectories.reshape(-1, *traj_start.shape[1:])
            trajectories = trajectories.to(self.accelerator.device)
            trajectories = self.models[fb].sample_trajectory(trajectories, self.prior)

            # Reduce number of timesteps for visualization
            num_timesteps = trajectories.shape[0]
            trajectories = torch.stack([
                trajectories[0], 
                trajectories[num_timesteps // 8], 
                trajectories[num_timesteps // 2], 
                trajectories[(num_timesteps * 7) // 8], 
                trajectories[-1]
            ], dim=0
            )

            if self.codec is not None:
                trajectories = self.codec.decode_to_image(trajectories.reshape(-1, *traj_start.shape[1:]))
                trajectories = trajectories.reshape(-1, self.num_trajectories * self.num_translations, *pred_x_start.shape[1:])

            visualize(
                exp_type=self.exp_type, 
                x_end=test_x_end, 
                x_start=test_x_start, 
                pred_x_start=pred_x_start, 
                fb=fb, 
                iteration=self.iteration, 
                exp_path=self.exp_path, 
                step=step,
                tracker=self.accelerator.get_tracker("wandb")
            )
            visualize_trajectory(
                exp_type=self.exp_type, 
                pred_x_start=pred_x_start, # type: ignore
                trajectories=trajectories,
                fb=fb,
                iteration=self.iteration, 
                exp_path=self.exp_path, 
                step=step,
                tracker=self.accelerator.get_tracker("wandb")
            )


    def eval(
        self,
        fb: Literal['forward', 'backward'],
        dataloader: DataLoader,
        step: Optional[int]
    ):
        self.fids[fb].reset()
        self.models[fb].eval()

        trange = tqdm(
            dataloader, 
            desc=f'{fb.capitalize()} D-IMF evaluation: {self.iteration}', 
            file=sys.stdout, 
            disable=not self.accelerator.is_local_main_process
        )
        with self.emas[fb].average_parameters():
            for test_x_start, test_x_end in trange:
                if self.codec is not None:
                    encoded_test_x_end = self.codec.encode_to_cats(test_x_end)
                    pred_x_start = self.models[fb].sample(encoded_test_x_end, self.prior)
                    pred_x_start = self.codec.decode_to_image(pred_x_start)
                else:
                    pred_x_start = self.models[fb].sample(test_x_end, self.prior)

                if not self.is_generation_normalized:
                    test_x_start = test_x_start.to(dtype=torch.uint8)
                    pred_x_start = pred_x_start.to(dtype=torch.uint8)
                    
                self.fids[fb].update(test_x_start, real=True)
                self.fids[fb].update(pred_x_start, real=False)

        self.accelerator.log({f'{fb}_fid': self.fids[fb].compute().detach()}, step=step)


    def train(
        self, 
        train_size: int, 
        eval_size: int,
        coupling_type: Literal['independent', 'prior'],
        trainset_x: BaseDataset, 
        trainset_y: BaseDataset,
        testset_x: BaseDataset, 
        testset_y: BaseDataset
    ):
        self.accelerator.print('Start training!')
        forward_testset = CouplingDataset(testset_y, conditional=testset_x)
        forward_testloader = DataLoader(forward_testset, batch_size=eval_size)
        forward_testloader = self.accelerator.prepare(forward_testloader)

        backward_testset = CouplingDataset(testset_x, conditional=testset_y)
        backward_testloader = DataLoader(backward_testset, batch_size=eval_size)
        backward_testloader = self.accelerator.prepare(backward_testloader)

        self.step = 0
        for self.iteration in range(1, self.iterations + 1):
            eval_size = train_size * self.inner_iterations
            if self.iteration == 1:
                eval_size = eval_size * self.prior_iterations

            ######## Forward ########
            if self.iteration == 1:
                forward_trainset = CouplingDataset(
                    trainset_y, 
                    conditional=trainset_x, 
                    type=coupling_type,
                    prior=self.prior
                ) 
            else:
                forward_trainset = trainset_y
            
            forward_sampler = RandomSampler(forward_trainset, replacement=True, num_samples=eval_size)
            forward_trainloader = DataLoader(forward_trainset, sampler=forward_sampler, batch_size=train_size, num_workers=8)
            forward_trainloader = self.accelerator.prepare(forward_trainloader)
            self.markovian_projection('forward', trainloader=forward_trainloader, testloader=forward_testloader)                
            with self.emas['forward'].average_parameters():
                self.accelerator.print(f'Saving chekpoint to {self.checkpoint_path}')
                self.accelerator.save_model(
                    self.models['forward'], 
                    os.path.join(self.checkpoint_path, f'forward_{self.iteration}')
                )

            ######## Backward ########
            if self.iteration == 1:
                backward_trainset = CouplingDataset(
                    trainset_x, 
                    conditional=trainset_y, 
                    type=coupling_type,
                    prior=self.prior
                ) 
            else:
                backward_trainset = trainset_x
            backward_sampler = RandomSampler(backward_trainset, replacement=True, num_samples=eval_size)
            backward_trainloader = DataLoader(backward_trainset, sampler=backward_sampler, batch_size=train_size, num_workers=8)
            backward_trainloader = self.accelerator.prepare(backward_trainloader)
            self.markovian_projection('backward', trainloader=backward_trainloader, testloader=backward_testloader)
            with self.emas['backward'].average_parameters():
                self.accelerator.print(f'Saving chekpoint to {self.checkpoint_path}')
                self.accelerator.save_model(
                    self.models['backward'], 
                    os.path.join(self.checkpoint_path, f'backward_{self.iteration}')
                )
        self.accelerator.print('End training!')

    def _get_map(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        a, b = ot.unif(x.shape[0]), ot.unif(y.shape[0])
        if self.codec is not None and self.prior.prior_type == 'centoid_gaussian':
            x = self.codec.centroids[x.long()]
            y = self.codec.centroids[y.long()]
        if x.dim() > 2:
            x = x.reshape(x.shape[0], -1)
        if y.dim() > 2:
            y = y.reshape(y.shape[0], -1)
        y = y.reshape(y.shape[0], -1)
        M = torch.cdist(x, y) ** 2
        p = ot.emd(a, b, M.detach().cpu().numpy())
        return p # type: ignore
    
    def _sample_map(self, pi: torch.Tensor, batch_size: int):
        p = pi.flatten()
        p = p / p.sum()
        choices = np.random.choice(pi.shape[0] * pi.shape[1], p=p, size=batch_size)
        return np.divmod(choices, pi.shape[1])
