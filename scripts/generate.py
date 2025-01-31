import argparse
import os
import sys
from omegaconf import OmegaConf
from tqdm.auto import tqdm

from accelerate import Accelerator
from accelerate.utils import set_seed
from safetensors.torch import load_file
from torchvision.transforms.functional import to_pil_image

set_seed(42)
from torch.utils.data import DataLoader

sys.path.append('../src')
from csbm.data import CelebaDataset, Prior
from csbm.models.quantized_images import Codec, LatentD3PM


def generate_images(
    accelerator: Accelerator, 
    dataloader: DataLoader, 
    codec: Codec, 
    model: LatentD3PM,
    prior: Prior,
    save_path: str,
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
    parser.add_argument('--exp_path', type=str, required=True)
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--iterations', nargs='*', type=int)
    args = parser.parse_args()

    # Config 
    config = OmegaConf.load(os.path.join(args.exp_path, 'config.yaml'))

    # Checkpoints
    checkpoints: str = os.path.join(args.exp_path, 'checkpoints')
    accelerator.print(f'Checkpoints path: {checkpoints}')
    if args.iterations is None:
        last_iteration = 0
        for file in os.listdir(checkpoints):
            if os.path.isdir(os.path.join(checkpoints, file)) and file.startswith('forward'):
                last_iteration += 1
        accelerator.print(f'Found {last_iteration} checkpoints!')     
        iter_range = range(1, last_iteration + 1)
    else:
        iter_range = args.iterations 

    # Data parms
    data_dir: str = args.data_dir
    batch_size: int = args.batch_size

    # Models
    accelerator.print('Init models...')
    codec = Codec(
        config_path=config.codec.config_path,
        ckpt_path=config.codec.ckpt_path,     
    ).to(accelerator.device)
    accelerator.print(f'Loaded model with {sum(p.numel() for p in codec.parameters()) / 1e6:3f} mil parameters!')
    codec.eval()

    # Data
    prior = Prior(
        alpha=config.prior.alpha, 
        num_categories=config.data.num_categories, 
        num_timesteps=config.data.num_timesteps, 
        num_skip_steps=config.data.num_skip_steps, 
        prior_type=config.prior.type
    ).to(accelerator.device)
    
    accelerator.print('Init dataset...')
    with accelerator.main_process_first(): # Avoid creating dirs at the same time
        testset = CelebaDataset(sex='male', size=config.data.dim, data_dir=data_dir, train=False, return_names=True)
    dataloader = DataLoader(testset, batch_size=batch_size)
    dataloader = accelerator.prepare(dataloader)

    accelerator.print('Starting generation...')
    for iteration in iter_range:
        iteration_path = os.path.join(checkpoints, f'forward_{iteration}')        
        model = LatentD3PM(
            input_dim=config.data.latent_dim,
            num_categories=config.data.num_categories, 
            num_timesteps=config.data.num_timesteps,
            **OmegaConf.to_object(config.model) # type: ignore
        )
        checkpoint = load_file(os.path.join(iteration_path, 'model.safetensors'))
        new_state_dict = {}
        for key, value in checkpoint.items():  
            new_key = key.replace("module.", "") 
            new_state_dict[new_key] = value

        model.load_state_dict(new_state_dict)
        model = model.eval()
        model.model = accelerator.prepare(model.model)
        accelerator.print(f'Loaded model with {sum(p.numel() for p in model.parameters()) / 1e6:3f} mil parameters!')

        accelerator.print(f'Generatig in {iteration_path}...')
        save_path = os.path.join(iteration_path, 'generation')
        if accelerator.is_main_process:
            os.makedirs(save_path, exist_ok=False)
        accelerator.wait_for_everyone()
        generate_images(accelerator, dataloader, codec, model, prior, save_path)

        del model
            
            
        
                
    
