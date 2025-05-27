import argparse
import os
import sys
from omegaconf import OmegaConf

from accelerate import Accelerator
from accelerate.utils import set_seed
from safetensors.torch import load_file
from transformers import PreTrainedTokenizerFast

set_seed(42)
import torch
from torch.utils.data import DataLoader

sys.path.append('src')
from csbm.data import (
    AmazonDataset,
    YelpDataset,
    AFHQDataset,
    CelebaDataset,
    DiscreteColoredMNISTDataset, 
    Prior,
    CouplingDataset
)
from csbm.models.images import ImageD3PM
from csbm.models.quantized_images import Codec, LatentD3PM
from csbm.models.texts import TextD3PM
from csbm.trainer import СSBMTrainer
from csbm.utils import ConsoleTracker


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_path', type=str, required=True)
    parser.add_argument('--iteration', type=int, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    args = parser.parse_args()
    exp_path = args.exp_path
    iteration = args.iteration
    data_dir = args.data_dir
    args = OmegaConf.load(os.path.join(exp_path, 'config.yaml'))

    accelerator = Accelerator(log_with=ConsoleTracker())
    accelerator.init_trackers('csbm', config=OmegaConf.to_object(args)) # type: ignore
    accelerator.print(f'Evaluating experiment with:\n{OmegaConf.to_yaml(args)}')    

    tokenizer = None
    if args.data.type == 'texts':
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=args.tokenizer.path,
            padding_side='right',
            truncation_side='right',
            bos_token='<s>',
            pad_token='<pad>',
        )

    accelerator.print('Loading dataset...')
    with accelerator.main_process_first(): # Avoid creating dirs at the same time
        if args.data.dataset == 'cmnist':
            testset_x = DiscreteColoredMNISTDataset(target_digit=3, data_dir=data_dir, train=False)
            testset_y = DiscreteColoredMNISTDataset(target_digit=2, data_dir=data_dir, train=False)
        elif args.data.dataset == 'celeba':
            # Train set is already quantized, so, we do not set size explicitly
            testset_x = CelebaDataset(sex='male', use_quantized=False, size=args.data.dim, data_dir=data_dir, train=False, split=args.data.train_test_split)
            testset_y = CelebaDataset(sex='female', use_quantized=False, size=args.data.dim, data_dir=data_dir, train=False, split=args.data.train_test_split)
        elif args.data.dataset == 'afhq':
            # Train set is already quantized, so, we do not set size explicitly
            testset_x = AFHQDataset(animal_type='cat', use_quantized=False, size=args.data.dim, data_dir=data_dir, train=False)
            testset_y = AFHQDataset(animal_type='wild', use_quantized=False, size=args.data.dim, data_dir=data_dir, train=False)
        elif args.data.dataset == 'yelp':
            assert tokenizer is not None, 'Tokenizer is not initialized!'
            testset_x = YelpDataset(sentiment='negative', data_dir=data_dir, tokenizer=tokenizer, max_length=args.data.dim, split='test')
            testset_y = YelpDataset(sentiment='positive', data_dir=data_dir, tokenizer=tokenizer, max_length=args.data.dim, split='test')
        elif args.data.dataset == 'amazon':
            assert tokenizer is not None, 'Tokenizer is not initialized!'
            testset_x = AmazonDataset(sentiment='negative', data_dir=data_dir, tokenizer=tokenizer, max_length=args.data.dim, split='test')
            testset_y = AmazonDataset(sentiment='positive', data_dir=data_dir, tokenizer=tokenizer, max_length=args.data.dim, split='test')
        else:
            raise NotImplementedError(f"The evaluation is not implemented for {args.data.type} exp!")

    codec = None
    centroids = None
    if args.data.type == 'quantized_images':
        codec = Codec(
            config_path=args.codec.config_path,
            ckpt_path=args.codec.ckpt_path,     
        )
        accelerator.print(f'Number of parameters in codec: {sum(p.numel() for p in codec.parameters())}')
        if args.prior.type == 'centroid_gaussian':
            centroids = codec.centroids

    accelerator.print('Loading prior...')
    prior = Prior(
        alpha=args.prior.alpha, 
        num_categories=args.data.num_categories if tokenizer is None else len(tokenizer), 
        num_timesteps=args.data.num_timesteps, 
        num_skip_steps=args.data.num_skip_steps, 
        prior_type=args.prior.type,
        eps=args.prior.eps,
        centroids=centroids if centroids is not None else None,
        dtype=torch.bfloat16 if args.train.low_precision else torch.float32,
    )
    if args.train.low_precision:
        prior = prior.bfloat16()

    accelerator.print('Loading models...')
    if args.data.type == 'images':
        model_class = ImageD3PM
    elif args.data.type == 'quantized_images':
        model_class = LatentD3PM
    elif args.data.type == 'texts':
        model_class = TextD3PM
    else:
        raise NotImplementedError(f"The evaluation is not implemented for {args.data.type} exp!")
    
    accelerator.print(f'Loading model on iteration {iteration}...')
    model = model_class(
        input_dim=args.data.dim if args.data.type != 'quantized_images' else args.data.latent_dim,
        num_categories=args.data.num_categories if tokenizer is None else len(tokenizer), 
        num_timesteps=args.data.num_timesteps,
        **OmegaConf.to_object(args.model) # type: ignore
    )
    accelerator.print(f'Number of parameters in model: {sum(p.numel() for p in model.parameters())}')
    checkpoint_path = os.path.join(
        exp_path, 'checkpoints', f'forward_{iteration}', 'model.safetensors'
    )
    checkpoint = load_file(checkpoint_path)
    new_state_dict = {}
    for key, value in checkpoint.items():  
        new_key = key.replace("module.", "") 
        new_state_dict[new_key] = value
    model.load_state_dict(new_state_dict)

    if codec is not None:
        codec = codec.to(accelerator.device)
    model.model = accelerator.prepare(
        model.model
    )
    prior = prior.to(accelerator.device)
    
    trainer = СSBMTrainer(
        iterations=args.train.iterations,
        inner_iterations=args.train.inner_iterations,
        prior_iterations=args.train.prior_iterations,
        use_mini_batch=args.train.use_mini_batch,
        accelerator=accelerator,
        forward_model=model,
        backward_model=None,
        prior=prior,
        codec=codec,
        tokenizer=tokenizer,
        forward_optimizer=None,
        backward_optimizer=None,
        forward_scheduler=None,
        backward_scheduler=None,
        kl_loss_coeff=args.train.kl_loss_coeff,
        ce_loss_coeff=args.train.ce_loss_coeff,
        mse_loss_coeff=args.train.mse_loss_coeff,
        ema_decay=args.train.ema_decay,
        eval_only=True,
        exp_type=args.data.type,
        exp_path=exp_path,
        eval_freq=args.eval.freq,
        num_trajectories=args.eval.num_trajectories,
        num_translations=args.eval.num_translations,
    )
    trainer.iteration = iteration

    testset = CouplingDataset(testset_y, conditional=testset_x)
    testloader = DataLoader(testset, batch_size=args.train.batch_size)
    testloader = accelerator.prepare(testloader)

    trainer.eval(
        'forward',
        testloader,
        step=iteration,
    )
    accelerator.end_training()
