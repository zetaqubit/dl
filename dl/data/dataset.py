"""Dataset and loaders for reading from prepared data.
"""
import os

import gin
import numpy as np
import pandas as pd
import torch

# Important for this to be on SSD so it's fast for memmap.
DATA_ROOT = '/home/z/data/zetaqubit/dl/data'

@gin.configurable
class MemoryMappedDataset(torch.utils.data.Dataset):
  def __init__(self, name, split, block_size, dbg_num_blocks=None,
               padding_token=None, min_nonpadding_tokens=64,
               align_to_seq_start=False):
    path = f'{DATA_ROOT}/{name}/{split}.bin'
    self.data = np.memmap(path, dtype=np.uint16, mode='r')
    self.block_size = block_size
    self.min_nonpadding_tokens = min(min_nonpadding_tokens, block_size)
    self.dataset_size = len(self.data)
    self.padding_token = padding_token
    if dbg_num_blocks is not None:
      self.dataset_size = min(self.dataset_size,
                              int(dbg_num_blocks * self.block_size) + 1)
    if align_to_seq_start:
      offsets = list(pd.read_csv(f'{path}.meta')['example_offset'])
      offsets = list(filter(lambda x: x <= self.dataset_size - self.block_size,
                            offsets))
      self.seq_offsets = offsets
    else:
      self.seq_offsets = None

  def __len__(self):
    return self.dataset_size

  def __getitem__(self, _):
    # Break contract and return a random span.
    def random_span():
      if self.seq_offsets:
        ix = torch.randint(len(self.seq_offsets), (1,))
        ix = self.seq_offsets[ix]
      else:
        ix = torch.randint(self.dataset_size - self.block_size, (1,))
      x = self.data[ix:ix+self.block_size]
      return x
    x = random_span()

    # Mark as padding all tokens to the right of the first padding token.
    if self.padding_token is not None:
      padding_mask = x == self.padding_token

      # Attempt to get a sequence with enough non-padding tokens.
      attempt = 0
      while ((idx := np.argmax(padding_mask)) != 0 and
             idx < self.min_nonpadding_tokens and attempt < 5):
        x = random_span()
        padding_mask = x == self.padding_token
        attempt += 1

      padding_mask = np.clip(np.cumsum(padding_mask), 0, 1)
      x = np.where(padding_mask, self.padding_token, x)

    # pin array x, which allows us to move them to GPU asynchronously (non_blocking=True)
    x = torch.from_numpy(x.astype(np.int64))
    x = x.pin_memory().to('cuda', non_blocking=True)
    return x
