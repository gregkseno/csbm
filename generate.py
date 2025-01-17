import argparse
import os
import sys
from omegaconf import OmegaConf
from tqdm.auto import tqdm

from accelerate import Accelerator, load_checkpoint_in_model
from accelerate.utils import set_seed
import torch
from torchvision.transforms.functional import to_pil_image

set_seed(42)
from torch.utils.data import DataLoader

sys.path.append('./src')
from dasbm.data import CelebaDataset, Prior
from dasbm.models.quantized_images import Codec, LatentD3PM


def generate_images(
    accelerator: Accelerator, 
    dataloader: DataLoader, 
    codec: Codec, 
    model: LatentD3PM,
    prior: Prior,
    save_path: str
):
    for images, image_names in tqdm(dataloader, file=sys.stdout, disable=(not accelerator.is_main_process)):
        images = images.to(accelerator.device) # type: ignore
        encoded_test_x_end = codec.encode_to_cats(images) # type: ignore
        predicts = model.sample(encoded_test_x_end, prior)
        predicts = codec.decode_to_image(predicts).cpu()
        for i, img in enumerate(predicts):
            img = to_pil_image(img)
            img.save(os.path.join(save_path, image_names[i])) # type: ignore


if __name__ == '__main__':
    set_seed(42)
    accelerator = Accelerator()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--exp_path', type=str, required=True)
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_timesteps', type=list, default=[1, 2, 4, 10, 25, 50, 75, 100])
    args = parser.parse_args()
    config = OmegaConf.load(args.config)

    # Checkpoints
    checkpoints: str = os.path.join(args.exp_path, 'checkpoints')
    accelerator.print(f'Checkpoints path: {checkpoints}')
    last_iteration = 0
    for file in os.listdir(checkpoints):
        if os.path.isdir(os.path.join(checkpoints, file)) and file.startswith('forward'):
            last_iteration += 1
    accelerator.print(f'Found {last_iteration} checkpoints!')

    # Data parms
    data_dir: str = args.data_dir
    batch_size: int = args.batch_size

    # Prior params
    default_num_timesteps = config.data.num_timesteps
    default_num_skip_steps = config.data.num_skip_steps
    num_timesteps: list[int] = sorted(args.num_timesteps, reverse=True)
    num_skip_steps: list[int] = [default_num_timesteps // num for num in num_timesteps]
    if default_num_timesteps not in num_timesteps:
        num_timesteps = num_timesteps + [default_num_timesteps]
        num_skip_steps = [default_num_skip_steps] + num_skip_steps

    # Models
    accelerator.print('Init models...')
    codec = Codec(
        config_path=config.codec.config_path,
        ckpt_path=config.codec.ckpt_path,     
    ).to(accelerator.device)
    accelerator.print(f'Loaded model with {sum(p.numel() for p in codec.parameters()) / 1e6:3f} mil parameters!')
    codec.eval()

    # Data
    accelerator.print('Init dataset...')
    with accelerator.main_process_first(): # Avoid creating dirs at the same time
        testset = CelebaDataset(sex='male', size=config.data.dim, data_dir=data_dir, train=False, return_names=True)
    dataloader = DataLoader(testset, batch_size=batch_size)

    dataloader = accelerator.prepare(dataloader)

    accelerator.print('Starting generation...')
    for iteration in range(1, last_iteration + 1):
        iteration_path = os.path.join(checkpoints, f'forward_{iteration}')        
        model = LatentD3PM(
            input_dim=config.data.latent_dim,
            num_categories=config.data.num_categories, 
            num_timesteps=config.data.num_timesteps,
            **OmegaConf.to_object(config.model) # type: ignore
        )
        accelerator.print(f'Loaded model with {sum(p.numel() for p in model.parameters()) / 1e6:3f} mil parameters!')

        model.model = accelerator.prepare(model.model)
        load_checkpoint_in_model(
            model, os.path.join(iteration_path, 'model.safetensors'), strict=True
        )
        model.eval()

        for timesteps, skip_steps in zip(num_timesteps, num_skip_steps):
            accelerator.print(f'Generatig with {timesteps} timesteps in {iteration_path}...')
            if timesteps != default_num_timesteps and iteration != last_iteration:
                accelerator.print(f'Skipping...')
                break
            save_path = os.path.join(iteration_path, f'steps_{timesteps}')
            if accelerator.is_main_process:
                os.makedirs(save_path, exist_ok=False)
            accelerator.wait_for_everyone()

            prior = Prior(
                alpha=config.prior.alpha, 
                num_categories=config.data.num_categories, 
                num_timesteps=timesteps, 
                num_skip_steps=skip_steps, 
                prior_type=config.prior.type
            ).to(accelerator.device)
            generate_images(accelerator, dataloader, codec, model, prior, save_path)
            del model
            
            
        
                
    
