import argparse
from typing import Literal, Optional
import json

import logging
import torch_geometric
from tqdm.contrib.logging import logging_redirect_tqdm
from tqdm.auto import tqdm
import wandb

import torch
from torch.optim import Optimizer # type: ignore
from torch.nn.utils import clip_grad_norm_

from dasbm.models import D3PM, DiffusionModel
from dasbm.data import DiscreteGaussianSampler, DiscreteSwissRollSampler, Prior, Sampler, MoleculeDatasetSampler
from dasbm.utils import create_expertiment, visualize, visualize_trajectory


class DiscreteSBMTrainer:
    forward_and_backward = {'forward', 'backward'}
    def __init__(
        self,
        iterations: int,
        inner_iterations: int,
        forward_model: D3PM,
        backward_model: D3PM,
        forward_optimizer: Optimizer,
        backward_optimizer: Optimizer,
        exp_path: str,
        ce_loss_coeff: float = 0.001,
        eval_freq: int = 1000,
    ) -> None:
        self.logger = logging.getLogger('DiscreteSBMTrainer')

        self.iterations = iterations
        self.inner_iterations = inner_iterations

        self.models = {
            'forward': forward_model,
            'backward': backward_model
        }
        self.optimizers = {
            'forward': forward_optimizer,
            'backward': backward_optimizer
        }
        self.ce_loss_coeff = ce_loss_coeff

        self.exp_path = exp_path
        self.eval_freq = eval_freq

    def markovian_projection(
        self,
        fb: Literal['forward', 'backward'],
        batch_size: int,
        sampler_start: Sampler, 
        sampler_end: Sampler
    ):  
        bf = 'backward' if fb == 'forward' else 'forward'
        self.models[fb].train()
        self.models[bf].eval()
        for _ in tqdm(range(self.inner_iterations), desc=f'{fb.capitalize()} D-IMF iteration: {self.iteration}'):
            self.step += 1
            self.optimizers[fb].zero_grad()
            true_x_start = sampler_start.sample(batch_size).to(self.models[fb].device)
            t = torch.randint(1, self.models[fb].num_timesteps + 2, (true_x_start.shape[0],), device=self.models[fb].device)

            if self.iteration == 1:
                x_t = self.models[fb].prior.sample(true_x_start, t)
            else:
                trajectory = self.models[bf].sample_trajectory(true_x_start)
                x_t = trajectory[t, torch.arange(batch_size)]

            
            # KL-divergence calculation
            pred_x_start_logits = self.models[fb](x_t, t)
            true_q_posterior_logits = self.models[fb].prior.q_posterior_logits(true_x_start, x_t, t) # .flatten(start_dim=0, end_dim=-2)
            pred_q_posterior_logits = self.models[fb].prior.q_posterior_logits(pred_x_start_logits, x_t, t) # .flatten(start_dim=0, end_dim=-2)
    
            kl_loss = torch.softmax(true_q_posterior_logits + self.models[fb].eps, dim=-1) * (
                torch.log_softmax(true_q_posterior_logits + self.models[fb].eps, dim=-1)
                - torch.log_softmax(pred_q_posterior_logits + self.models[fb].eps, dim=-1)
            )
            kl_loss = kl_loss.sum(dim=-1).mean()
            # Cross-Entropy calculation
            pred_x_start_logits = pred_x_start_logits.flatten(start_dim=0, end_dim=-2)
            true_x_start = true_x_start.flatten(start_dim=0, end_dim=-1)
            
            ce_loss = torch.nn.CrossEntropyLoss()(pred_x_start_logits, true_x_start)
    
            loss = kl_loss + self.ce_loss_coeff * ce_loss
            info = {
                "kl_loss": kl_loss.detach().item(),
                "ce_loss": ce_loss.detach().item(),
            }
            loss.backward()
            # clip_grad_norm_(self.models[fb].parameters(), 0.01)
            self.optimizers[fb].step()
            
            if self.step % self.eval_freq == 0:
                with logging_redirect_tqdm():
                    self.logger.info(f'{fb.capitalize()} D-IMF iteration: {self.iteration}: {info["kl_loss"]}, ce_loss: {info["ce_loss"]}')
                self.eval(
                    fb, 1024,
                    sampler_start=sampler_start,
                    sampler_end=sampler_end,
                    step=self.step
                )
            if wandb.run:
                wandb.log(info, step=self.step)
        
    def eval(
        self,
        fb: Literal['forward', 'backward'],
        batch_size: int,
        sampler_start: Sampler, 
        sampler_end: Sampler, 
        step: Optional[int]
    ):
        self.models[fb].eval()
        test_x_end = sampler_end.sample(batch_size).to(self.models[fb].device)
        test_x_start = sampler_start.sample(batch_size)
        pred_x_start = self.models[fb].sample(test_x_end)
        visualize(test_x_end, test_x_start, pred_x_start, fb, iteration=self.iteration, exp_path=self.exp_path, step=step)
        visualize_trajectory(test_x_end, self.models[fb], fb, iteration=self.iteration, exp_path=self.exp_path, step=step)

    def train(self, batch_size: int, sampler_x: Sampler, sampler_y: Sampler):
        self.logger.info('Start training!')
        self.step = 0
        for self.iteration in range(1, self.iterations + 1):
            ######## Forward ########
            self.markovian_projection(
                'forward', batch_size, 
                sampler_start=sampler_y,
                sampler_end=sampler_x
            )                

            ######## Backward ########
            self.markovian_projection(
                'backward', batch_size, 
                sampler_start=sampler_x,
                sampler_end=sampler_y 
            )
         
        self.logger.info('End training!')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('Train script')

    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_type', choices=['toy', 'graphs'], type=str)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--input_dim', type=int, default=2)
    parser.add_argument('--num_categories', type=int, default=50)
    parser.add_argument('--num_timesteps', type=int, default=100)
    parser.add_argument('--prior', choices=['random_jump', 'random_neighbour', 'd3pm'], type=str)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--iterations', type=int, default=5)
    parser.add_argument('--inner_iterations', type=int, default=7_000)
    parser.add_argument('--lr', type=float, default=4e-4)
    parser.add_argument('--ce_loss_coeff', type=float, default=0.001)
    parser.add_argument('--eval_freq', type=int, default=1000)
    args = parser.parse_args()

    exp_name, exp_path = create_expertiment(args)
    wandb.init(project='Discrete SBM', name=exp_name, config=vars(args))
    logger.info(f'Created experiment folder in {exp_path}.')
    logger.info(f'Initializing models with:\n{json.dumps(vars(args), indent=4)}')
    if torch.cuda.is_available():
        device = 'cuda'
        logger.info('GPU is found :)')
    else:
        device = 'cpu'
        logger.info('GPU is NOT found :(')    

    prior = Prior(
        alpha=args.alpha, 
        num_categories=args.num_categories, 
        num_timesteps=args.num_timesteps, 
        prior_type=args.prior
    )
    forward_d3pm = D3PM(
        DiffusionModel(
            args.input_dim, 
            args.num_categories, 
            args.num_timesteps
        ), 
        prior, 
    ).to(device)

    backward_d3pm = D3PM(
        DiffusionModel(
            args.input_dim, 
            args.num_categories, 
            args.num_timesteps
        ), 
        prior, 
    ).to(device)

    forward_optimizer = torch.optim.AdamW(forward_d3pm.parameters(), lr=args.lr, betas=(0.95, 0.99)) # type: ignore
    backward_optimizer = torch.optim.AdamW(backward_d3pm.parameters(), lr=args.lr, betas=(0.95, 0.99)) # type: ignore

    if args.exp_type == 'toy':
        sampler_x = DiscreteGaussianSampler(dim=args.input_dim, num_categories=args.num_categories)
        sampler_y = DiscreteSwissRollSampler(num_categories=args.num_categories)
    else:
        sampler_x = MoleculeDatasetSampler(torch_geometric.datasets.MoleculeNet(root='../data/', name='FreeSolv'))
        sampler_y = MoleculeDatasetSampler(torch_geometric.datasets.MoleculeNet(root='../data/', name='FreeSolv'))

    trainer = DiscreteSBMTrainer(
        iterations=args.iterations,
        inner_iterations=args.inner_iterations,
        forward_model=forward_d3pm,
        backward_model=backward_d3pm,
        forward_optimizer=forward_optimizer,
        backward_optimizer=backward_optimizer,
        ce_loss_coeff=args.ce_loss_coeff,
        exp_path=exp_path,
        eval_freq=args.eval_freq,
    )
    trainer.train(args.batch_size, sampler_x, sampler_y)
