'''Prepares data for training - downloading, splitting, tokenizing.

Due to quirks of the huggingface dataset preprocessing and how it tries to
pickle the map function, the main part of this script cannot be in a function.

To read the bin files later, e.g. with numpy:
  m = np.memmap('train.bin', dtype=np.uint16, mode='r')
'''

import os

from absl import app
from absl import flags
from datasets import load_dataset  # huggingface datasets
import numpy as np
from tqdm import tqdm

from dl.data import tokenizers

flags.DEFINE_string('dataset', None, 'Dataset to download.')
flags.DEFINE_string('tokenizer', None, 'Type of tokenizer to use.')

FLAGS = flags.FLAGS

# number of workers in .map() call
# good number to use is ~order number of cpu cores // 2
_NUM_PROC = 24


def prepare(dataset, tok_type):
  if dataset in ('openwebtext') and tok_type == 'char':
    print(f'{dataset} + {tok_type} combination too large. Skipping.')
    return

  if dataset == 'wikitext-103':
    ds = load_dataset(path='wikitext', name='wikitext-103-v1')
  else:
    ds = load_dataset(dataset)

  if dataset in ('wikitext-103'):
    # These datasets have a validation split, so just rename it.
    ds['val'] = ds.pop('validation')  # rename
  elif dataset in ('enwik8'):
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


  tokenizer = tokenizers.create(tok_type, max_seq_len=None)
  assert tokenizer.vocab_size < 2**16, 'Max token exceeds uint16'

  def process(example):
      ids = tokenizer.encode(example['text']) # encode_ordinary ignores any special tokens
      ids.append(tokenizer.padding_id) # add the end of text token, e.g. 50256 for gpt2 bpe
      # note: I think eot should be prepended not appended... hmm. it's called 'eot' though...
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
      idx = 0
      for example in tqdm(dset):
          arr[idx : idx + example['len']] = example['ids']
          idx += example['len']
      arr.flush()


def main(_):
  datasets = [FLAGS.dataset] if FLAGS.dataset else [
    'enwik8', 'wikitext-103', 'openwebtext',
  ]
  tokenizers = [FLAGS.tokenizer] if FLAGS.tokenizer else [
    'char', 'gpt2',
  ]
  for dataset in datasets:
    for tokenizer in tokenizers:
      prepare(dataset, tokenizer)


if __name__ == '__main__':
  app.run(main)
