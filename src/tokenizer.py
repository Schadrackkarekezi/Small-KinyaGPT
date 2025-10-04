
import argparse
from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, normalizers

def train_bpe_tokenizer(dataset_name: str, vocab_size: int, save_path: str):
    """Train and save a BPE tokenizer from a Hugging Face dataset"""
    dataset = load_dataset(dataset_name, split="train")

    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.normalizer = normalizers.NFKC()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=["<s>", "<pad>", "</s>", "<unk>"]
    )

    def batch_iterator(batch_size=1000):
        for i in range(0, len(dataset), batch_size):
            yield dataset[i: i + batch_size]["text"]

    tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)
    tokenizer.save(save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="mbazaNLP/kinyarwanda_monolingual_v01.1")
    parser.add_argument("--vocab_size", type=int, default=32000)
    parser.add_argument("--save_path", type=str, default="data/kinyarwanda_bpe.json")
    args = parser.parse_args()

    train_bpe_tokenizer(args.dataset, args.vocab_size, args.save_path)
