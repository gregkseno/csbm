import argparse
import os
import sys
from omegaconf import OmegaConf
from tqdm.auto import tqdm

from accelerate import Accelerator
from accelerate.utils import set_seed
from safetensors.torch import load_file
from torchvision.transforms.functional import to_pil_image
from transformers import PreTrainedTokenizerFast
import pandas as pd

set_seed(42)
from torch.utils.data import DataLoader

sys.path.append('../src')
from csbm.data import AFHQDataset, CelebaDataset, YelpDataset, Prior
from csbm.models.quantized_images import Codec, LatentD3PM
from csbm.models.texts import TextD3PM


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


def generate_texts(
    accelerator: Accelerator, 
    dataloader: DataLoader, 
    tokenizer: PreTrainedTokenizerFast, 
    model: TextD3PM,
    prior: Prior,
    save_path: str,
):
    dataframe = pd.DataFrame(columns=['text', 'predict'])
    for texts in tqdm(dataloader, file=sys.stdout, disable=(not accelerator.is_main_process)):
        texts = texts.to(accelerator.device) # type: ignore
        predicts = model.sample(texts, prior)

        texts = tokenizer.batch_decode(texts.cpu())
        predicts = tokenizer.batch_decode(predicts.cpu())
        dataframe = pd.concat(
            [dataframe, pd.DataFrame({'text': texts, 'predict': predicts})], ignore_index=True
        )
    dataframe.to_csv(os.path.join(save_path, 'texts.csv'), index=False)
        


if __name__ == '__main__':
    set_seed(42)
    accelerator = Accelerator()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_path', type=str, required=True)
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--iteration', type=int, required=True)
    args = parser.parse_args()

    # Config 
    config = OmegaConf.load(os.path.join(args.exp_path, 'config.yaml'))

    # Checkpoints
    checkpoints: str = os.path.join(args.exp_path, 'checkpoints')
    accelerator.print(f'Checkpoints path: {checkpoints}')
    iteration = args.iteration

    # Data parms
    data_dir: str = args.data_dir
    batch_size: int = args.batch_size

    # Models
    accelerator.print('Init models...')
    
    codec = None
    tokenizer = None
    if config.data.type == 'quantized_images':
        codec = Codec(
            config_path=config.codec.config_path,
            ckpt_path=config.codec.ckpt_path,     
        ).to(accelerator.device)
        codec.eval()
        accelerator.print(f'Loaded model with {sum(p.numel() for p in codec.parameters()) / 1e6:3f} mil parameters!')
    elif config.data.type == 'texts':
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=config.tokenizer.path,
            padding_side='right',
            truncation_side='right',
            bos_token='<s>',
            pad_token='<pad>',
        )

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
        if config.data.dataset == 'celeba':
            testset = CelebaDataset(
                sex='male', 
                use_quantized=False, 
                size=config.data.dim, 
                data_dir=data_dir, 
                train=False, 
                split=config.data.train_test_split,
                return_names=True
            )
        elif config.data.dataset == 'afhq':
            testset = AFHQDataset(
                animal_type='cat', 
                use_quantized=False, 
                size=config.data.dim, 
                data_dir=data_dir, 
                train=False,
                return_names=True
            )
        elif config.data.dataset == 'yelp':
            assert tokenizer is not None, 'Tokenizer is not initialized!'
            testset = YelpDataset(
                sentiment='negative', 
                data_dir=data_dir, 
                tokenizer=tokenizer, 
                max_length=args.data.dim, 
                split='eval'
            )
        else:
            raise NotImplementedError(f"Dataset {config.data.type} is required, but only `celeba`, `afhq` and `yelp` datasets are implemented!")

    dataloader = DataLoader(testset, batch_size=batch_size, num_workers=4, shuffle=False)
    dataloader = accelerator.prepare(dataloader)

    accelerator.print('Starting generation...')
    iteration_path = os.path.join(checkpoints, f'forward_{iteration}')   
    if config.data.type == 'quantized_images':
        model_class = LatentD3PM
    elif config.data.type == 'texts':
        model_class = TextD3PM
    else:
        raise NotImplementedError(f"Model {config.data.type} is required, but only `quantized_images` and `texts` experiments are implemented!")
    
    model = model_class(
        input_dim=config.data.dim if config.data.type != 'quantized_images' else config.data.latent_dim,
        num_categories=config.data.num_categories if tokenizer is None else len(tokenizer), 
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

    if config.data.type == 'quantized_images':
        assert codec is not None, 'Codec is not initialized!'
        assert isinstance(model, LatentD3PM), 'Model is not LatentD3PM!'
        generate_images(accelerator, dataloader, codec, model, prior, save_path)
    elif config.data.type == 'texts':
        assert tokenizer is not None, 'Tokenizer is not initialized!'
        assert isinstance(model, TextD3PM), 'Model is not TextD3PM!'
        generate_texts(accelerator, dataloader, tokenizer, model, prior, save_path)

    del model
            
            
        
                
    
