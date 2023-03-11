# Adapted from https://github.com/karpathy/nanoGPT
# saves the openwebtext dataset to a binary file for training. following was helpful:
# https://github.com/HazyResearch/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py

import os
import sys

from absl import flags
from datasets import load_dataset # huggingface datasets
import numpy as np
import gin
import tiktoken
from tqdm import tqdm

from dl.data import tokenizers
from dl.utils.config_utils import gin_get

flags.DEFINE_multi_string(
    'ginc', [],
    'List of config files.')
flags.DEFINE_multi_string(
    'ginp', [],
    'Newline separated list of Gin parameter bindings.')

flags.FLAGS(sys.argv)
FLAGS = flags.FLAGS


gin.parse_config_files_and_bindings(FLAGS.ginc, FLAGS.ginp)

# number of workers in .map() call
# good number to use is ~order number of cpu cores // 2
num_proc = 8

# takes 100MB in huggingface .cache dir, about 1M documents (1,128,024)
dataset = load_dataset("enwik8")

DATASET_DIR = '/home/z/data/zetaqubit/dl/data/enwik8/'

# owt by default only contains the 'train' split, so create a test split
split_dataset = dataset["train"].train_test_split(test_size=0.05, seed=2357, shuffle=True)
split_dataset['val'] = split_dataset.pop('test') # rename the test split to val

# this results in:
# >>> split_dataset
# DatasetDict({
#     train: Dataset({
#         features: ['text'],
#         num_rows: 1071622
#     })
#     val: Dataset({
#         features: ['text'],
#         num_rows: 56402
#     })
# })

# we now want to tokenize the dataset. first define the encoding function (gpt2 bpe)
# enc = tiktoken.get_encoding("gpt2")
tok_type = gin_get('%tok_type', 'gpt2')
tokenizer = tokenizers.create(tok_type)
def process(example):
    ids = tokenizer.encode(example['text']) # encode_ordinary ignores any special tokens
    ids.append(tokenizer.padding_id) # add the end of text token, e.g. 50256 for gpt2 bpe
    # note: I think eot should be prepended not appended... hmm. it's called "eot" though...
    out = {'ids': ids, 'len': len(ids)}
    return out

# tokenize the dataset
tokenized = split_dataset.map(
    process,
    remove_columns=['text'],
    desc="tokenizing the splits",
    num_proc=num_proc,
    load_from_cache_file=False,  # Caching not supported for some tokenizers.
)

os.makedirs(DATASET_DIR, exist_ok=True)

# concatenate all the ids in each dataset into one large file we can use for training
for split, dset in tokenized.items():
    arr_len = np.sum(dset['len'])
    filename = os.path.join(DATASET_DIR, f'{tok_type}.{split}.bin')
    dtype = np.uint16 # (can do since enc.max_token_value == 50256 is < 2**16)
    arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))

    print(f"writing {filename}...")
    idx = 0
    for example in tqdm(dset):
        arr[idx : idx + example['len']] = example['ids']
        idx += example['len']
    arr.flush()

# train.bin is ~17GB, val.bin ~8.5MB
# train has ~9B tokens (9,035,582,198)
# val has ~4M tokens (4,434,897)

# to read the bin files later, e.g. with numpy:
# m = np.memmap('train.bin', dtype=np.uint16, mode='r')

