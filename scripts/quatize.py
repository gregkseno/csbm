import argparse
import sys

from omegaconf import OmegaConf
import torch

sys.path.append('../src')
from dasbm.models.quantized_images import Codec
from dasbm.data import CelebaDataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    args = parser.parse_args()

    data_dir = args.data_dir
    args = OmegaConf.load(args.config)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device}!')

    codec = Codec(
        config_path=args.codec.config_path,
        ckpt_path=args.codec.ckpt_path,     
    ).to(device)

    print('Start quantization...')
    CelebaDataset.quantize_train(codec, data_dir, args.data.dim, args.train.batch_size)