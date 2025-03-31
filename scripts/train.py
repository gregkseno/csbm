import argparse
import os
import sys
from omegaconf import OmegaConf

from accelerate import Accelerator, ProfileKwargs
from accelerate.utils import set_seed

set_seed(42)
import torch

sys.path.append('src')
from csbm.data import (
    AFHQDataset,
    CelebaDataset,
    DiscreteGaussianDataset, 
    DiscreteSwissRollDataset, 
    DiscreteColoredMNISTDataset, 
    Prior
)
from csbm.models.toy import D3PM
from csbm.models.images import ImageD3PM
from csbm.models.quantized_images import Codec, LatentD3PM
from csbm.vq_diffusion.engine.lr_scheduler import ReduceLROnPlateauWithWarmup
from csbm.trainer import СSBMTrainer
from csbm.utils import create_expertiment


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--exp_dir', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    args = parser.parse_args()
    exp_dir = args.exp_dir
    data_dir = args.data_dir
    args = OmegaConf.load(args.config)

    accelerator = Accelerator(
        log_with="wandb", 
        cpu=False, 
        gradient_accumulation_steps=args.train.gradient_accumulation_steps
    )
    
    exp_name, exp_path = None, None
    if accelerator.is_main_process:
        exp_name, exp_path = create_expertiment(exp_dir, args)
        OmegaConf.save(config=args, f=os.path.join(exp_path, 'config.yaml'))
    
    accelerator.init_trackers(
        project_name='Discrete SBM', 
        init_kwargs={'wandb': {'name': exp_name}}, 
        config=OmegaConf.to_object(args) # type: ignore
    )
    accelerator.print(f'Created experiment folder in {exp_path}.')
    accelerator.print(f'Initializing experiment with:\n{OmegaConf.to_yaml(args)}')    

    accelerator.print('Loading dataset...')
    with accelerator.main_process_first(): # Avoid creating dirs at the same time
        if args.data.dataset == 'swiss_roll':
            n_samples = args.train.batch_size * args.train.inner_iterations
            trainset_x = DiscreteGaussianDataset(n_samples=n_samples, dim=args.data.dim, num_categories=args.data.num_categories)
            trainset_y = DiscreteSwissRollDataset(n_samples=n_samples, num_categories=args.data.num_categories)
            testset_x = DiscreteGaussianDataset(n_samples=n_samples, dim=args.data.dim, num_categories=args.data.num_categories, train=False)
            testset_y = DiscreteSwissRollDataset(n_samples=n_samples, num_categories=args.data.num_categories, train=False)
        elif args.data.dataset == 'cmnist':
            trainset_x = DiscreteColoredMNISTDataset(target_digit=3, data_dir=data_dir)
            trainset_y = DiscreteColoredMNISTDataset(target_digit=2, data_dir=data_dir)
            testset_x = DiscreteColoredMNISTDataset(target_digit=3, data_dir=data_dir, train=False)
            testset_y = DiscreteColoredMNISTDataset(target_digit=2, data_dir=data_dir, train=False)
            # Train set is already quantized, so, we do not set size explicitly
        elif args.data.dataset == 'celeba':
            trainset_x = CelebaDataset(sex='male', size=args.data.dim, data_dir=data_dir, split=args.data.train_test_split)
            trainset_y = CelebaDataset(sex='female', size=args.data.dim, data_dir=data_dir, split=args.data.train_test_split)
            testset_x = CelebaDataset(sex='male', use_quantized=False, size=args.data.dim, data_dir=data_dir, train=False, split=args.data.train_test_split)
            testset_y = CelebaDataset(sex='female', use_quantized=False, size=args.data.dim, data_dir=data_dir, train=False, split=args.data.train_test_split)
        elif args.data.dataset == 'afhq':
            trainset_x = AFHQDataset(animal_type='cat', size=args.data.dim, data_dir=data_dir)
            trainset_y = AFHQDataset(animal_type='wild', size=args.data.dim, data_dir=data_dir)
            testset_x = AFHQDataset(animal_type='cat', use_quantized=False, size=args.data.dim, data_dir=data_dir, train=False)
            testset_y = AFHQDataset(animal_type='wild', use_quantized=False, size=args.data.dim, data_dir=data_dir, train=False)
        else:
            raise NotImplementedError(f"Unknown exp type {args.data.type}!")

    codec = None
    if args.data.type == 'quantized_images':
        codec = Codec(
            config_path=args.codec.config_path,
            ckpt_path=args.codec.ckpt_path,     
        )

    prior = Prior(
        alpha=args.prior.alpha, 
        num_categories=args.data.num_categories, 
        num_timesteps=args.data.num_timesteps, 
        num_skip_steps=args.data.num_skip_steps, 
        prior_type=args.prior.type,
        centroids=codec.centroids if codec is not None and args.prior.type == 'centroid_gaussian' else None
    )

    if args.data.type == 'toy':
        model_class = D3PM
    elif args.data.type == 'images':
        model_class = ImageD3PM
    elif args.data.type == 'quantized_images':
        model_class = LatentD3PM
    else:
        raise NotImplementedError(f"Unknown exp type {args.data.type}!")
    
    forward_model = model_class(
        input_dim=args.data.dim if args.data.type != 'quantized_images' else args.data.latent_dim,
        num_categories=args.data.num_categories, 
        num_timesteps=args.data.num_timesteps,
        **OmegaConf.to_object(args.model) # type: ignore
    )
    backward_model = model_class(
        input_dim=args.data.dim if args.data.type != 'quantized_images' else args.data.latent_dim,
        num_categories=args.data.num_categories, 
        num_timesteps=args.data.num_timesteps,
        **OmegaConf.to_object(args.model) # type: ignore
    )

    forward_optimizer = torch.optim.AdamW(forward_model.parameters(), **args.train.optimizer) # type: ignore
    backward_optimizer = torch.optim.AdamW(backward_model.parameters(), **args.train.optimizer) # type: ignore
    
    forward_scheduler, backward_scheduler = None, None
    if 'scheduler' in args.train:
        forward_scheduler = ReduceLROnPlateauWithWarmup(forward_optimizer, **args.train.scheduler)
        backward_scheduler = ReduceLROnPlateauWithWarmup(backward_optimizer, **args.train.scheduler)
        forward_scheduler = accelerator.prepare(forward_scheduler)
        backward_scheduler = accelerator.prepare(backward_scheduler)

    if codec is not None:
        codec = codec.to(accelerator.device)
    forward_model.model, forward_optimizer = accelerator.prepare(
        forward_model.model, forward_optimizer
    )
    backward_model.model, backward_optimizer = accelerator.prepare(
        backward_model.model, backward_optimizer
    )
    prior = prior.to(accelerator.device)
    
    trainer = СSBMTrainer(
        iterations=args.train.iterations,
        inner_iterations=args.train.inner_iterations,
        prior_iterations=args.train.prior_iterations,
        use_mini_batch=args.train.use_mini_batch,
        accelerator=accelerator,
        forward_model=forward_model,
        backward_model=backward_model,
        prior=prior,
        codec=codec,
        forward_optimizer=forward_optimizer,
        backward_optimizer=backward_optimizer,
        forward_scheduler=forward_scheduler,
        backward_scheduler=backward_scheduler,
        kl_loss_coeff=args.train.kl_loss_coeff,
        ce_loss_coeff=args.train.ce_loss_coeff,
        mse_loss_coeff=args.train.mse_loss_coeff,
        ema_decay=args.train.ema_decay,
        exp_type=args.data.type,
        exp_path=exp_path,
        eval_freq=args.eval.freq,
        num_trajectories=args.eval.num_trajectories,
        num_translations=args.eval.num_translations,
    )

    trainer.train(
        train_size=args.train.batch_size, 
        eval_size=args.eval.num_samples,
        coupling_type=args.data.coupling_type,
        trainset_x=trainset_x, 
        trainset_y=trainset_y,
        testset_x=testset_x, 
        testset_y=testset_y
    )
    accelerator.end_training()
