import argparse
import sys

from omegaconf import OmegaConf

from tokenizers import Tokenizer
from tokenizers import models, pre_tokenizers, decoders, trainers

sys.path.append('src')
from csbm.data import YelpDataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    args = parser.parse_args()

    data_dir = args.data_dir
    print(f"Loading config from {args.config}")
    args = OmegaConf.load(args.config)

    # Load the dataset for training the tokenizer
    dataset = YelpDataset(sentiment='all', data_dir=data_dir, split=1.0)
    print(f"Loaded dataset with {len(dataset)} samples.")
    print(dataset[0])

    # Train the tokenizer
    # Configure the tokenizer parameters from config
    tokenizer = Tokenizer(models.Unigram()) # type: ignore
    tokenizer.pre_tokenizer = pre_tokenizers.Metaspace() # type: ignore
    tokenizer.decoder = decoders.Metaspace() # type: ignore

    trainer = trainers.UnigramTrainer(
        vocab_size=args.data.num_categories, 
        special_tokens=["<unk>", "<pad>"], 
        unk_token="<unk>",
        show_progress=True,
    )
    # print("Training tokenizer...")
    # tokenizer.train_from_iterator(
    #     (str(x) for x in dataset),
    #     trainer=trainer,
    # )
    
    # tokenizer.save("checkpoints/tokenizer-yelp.json")