import argparse
import sys

from omegaconf import OmegaConf

from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer
from tokenizers.models import Unigram
from tokenizers.pre_tokenizers import Metaspace
from tokenizers.trainers import UnigramTrainer
import torch

sys.path.append('src')
from csbm.data import YelpDataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    args = parser.parse_args()

    data_dir = args.data_dir
    args = OmegaConf.load(args.config)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device}!')

    # Load the dataset for training the tokenizer
    dataset = YelpDataset(sentiment='all', data_dir=data_dir, split=1.0)
    print(dataset[0])

    # Train the tokenizer
    # Configure the tokenizer parameters from config
    tokenizer = Tokenizer(Unigram()) # type: ignore
    tokenizer.pre_tokenizer = Metaspace() # type: ignore
    tokenizer.decoder = Metaspace() # type: ignore

    trainer = UnigramTrainer(
        vocab_size=args.data.num_categories, 
        special_tokens=["<unk>", "<pad>"], 
        unk_token="<unk>"
    )
    tokenizer.train_from_iterator(
        (str(x) for x in dataset),
        trainer=trainer,
    )
    
    tokenizer.save("data/tokenizer-yelp.json")