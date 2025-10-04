
from datasets import load_dataset

dataset = load_dataset("mbazaNLP/kinyarwanda_monolingual_v01.1", split="train")
print(dataset)
print(dataset[0])

