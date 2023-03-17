'''Prepares data for training - downloading, splitting, tokenizing.

To read the bin files later, e.g. with numpy:
  m = np.memmap('train.bin', dtype=np.uint16, mode='r')

Supported datasets:
  - enwik8
  - wikitext-103
  - openwebtext
  - ptb: Penn Treebank


Supported tokenizers:
  - char: ASCII character tokenizer (ignores non-ASCII)
  - gpt2
'''

import os

from absl import app
from absl import flags
import datasets as hf_datasets  # huggingface datasets
import numpy as np
import pandas as pd
from tqdm import tqdm

from dl.data import tokenizers

flags.DEFINE_string('dataset', None, 'Dataset to download.')
flags.DEFINE_string('tokenizer', None, 'Type of tokenizer to use.')

FLAGS = flags.FLAGS

# number of workers in .map() call
# good number to use is ~order number of cpu cores // 2
_NUM_PROC = 12


def prepare(dataset, tok_type):
  if dataset in ('openwebtext') and tok_type == 'char':
    print(f'{dataset} + {tok_type} combination too large. Skipping.')
    return

  tokenizer = tokenizers.create(tok_type, max_seq_len=None)
  assert tokenizer.vocab_size < 2**16, 'Max token exceeds uint16'

  sep_ids = [tokenizer.padding_id]
  if dataset == 'wikitext-103':
    ds = hf_datasets.load_dataset(path='wikitext', name='wikitext-103-v1')
    sep_ids = tokenizer.encode('\n')
  elif dataset == 'ptb':
    ds = hf_datasets.load_dataset('ptb_text_only')
    ds = ds.rename_column('sentence', 'text')
  elif dataset == 'shakespeare':
    df = pd.read_csv('dl/data/tiny_shakespeare.txt', delimiter='\t', header=0,
                     names=['text'])
    ds = hf_datasets.Dataset.from_pandas(df, split='train')
    ds = hf_datasets.DatasetDict({'train': ds})
    sep_ids = tokenizer.encode('\n')
  elif dataset == 'abc':
    abc = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
    ds = hf_datasets.DatasetDict({
      'train': hf_datasets.Dataset.from_dict({'text': [abc] * 1000}),
    })
    sep_ids = []
  else:
    ds = hf_datasets.load_dataset(dataset)

  if dataset in ('wikitext-103', 'ptb'):
    # These datasets have a validation split, so just rename it.
    ds['val'] = ds.pop('validation')  # rename
  elif dataset in ('abc', 'shakespeare', 'enwik8'):
    # These datasets don't have a validation split, so create it ourselves.
    ds = ds['train'].train_test_split(test_size=0.05, seed=2357, shuffle=True)
    ds['val'] = ds.pop('test')  # rename
  elif dataset in ('openwebtext'):
    # These datasets don't have a validation split, so create it ourselves.
    ds = ds['train'].train_test_split(test_size=0.0005, seed=2357, shuffle=True)
    ds['val'] = ds.pop('test')  # rename
  assert 'train' in ds
  assert 'val' in ds

  dataset_dir = f'/home/z/data/zetaqubit/dl/data/{dataset}/'
  os.makedirs(dataset_dir, exist_ok=True)


  def process(example):
    ids = tokenizer.encode(example['text'])
    ids.extend(sep_ids)
    out = {'ids': ids, 'len': len(ids)}
    return out

  # tokenize the dataset
  tokenized = ds.map(
      process,
      remove_columns=['text'],
      desc='tokenizing the splits',
      num_proc=_NUM_PROC,
  )

  # concatenate all the ids in each dataset into one large file we can use for training
  for split, dset in tokenized.items():
    arr_len = np.sum(dset['len'])
    filename = os.path.join(dataset_dir, f'{tok_type}.{split}.bin')
    dtype = np.uint16
    arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))

    print(f'writing {filename}...')
    metadata_rows = []
    idx = 0
    for example in tqdm(dset):
      metadata = {'example_offset': idx}
      arr[idx : idx + example['len']] = example['ids']
      idx += example['len']
      metadata_rows.append(metadata)
    arr.flush()
    df_metadata = pd.DataFrame(metadata_rows)
    df_metadata.to_csv(filename + '.meta', index=False)


def main(_):
  datasets = [FLAGS.dataset] if FLAGS.dataset else [
    'abc', 'shakespeare', 'enwik8', 'wikitext-103', 'openwebtext',
  ]
  tokenizers = [FLAGS.tokenizer] if FLAGS.tokenizer else [
    'char', 'gpt2',
  ]
  for dataset in datasets:
    for tokenizer in tokenizers:
      prepare(dataset, tokenizer)


if __name__ == '__main__':
  app.run(main)
