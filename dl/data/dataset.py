"""Dataset and loaders for reading from prepared data.
"""
import os

import gin
import numpy as np
import torch

# Important for this to be on SSD so it's fast for memmap.
DATA_ROOT = '/home/z/data/zetaqubit/dl/data'

@gin.configurable
class MemoryMappedDataset(torch.utils.data.Dataset):
  def __init__(self, name, split, block_size, dbg_num_blocks=None):
    path = f'{DATA_ROOT}/{name}/{split}.bin'
    self.data = np.memmap(path, dtype=np.uint16, mode='r')
    self.block_size = block_size
    self.dataset_size = len(self.data)
    if dbg_num_blocks is not None:
      self.dataset_size = min(self.dataset_size,
                              int(dbg_num_blocks * self.block_size) + 1)

  def __len__(self):
    return self.dataset_size

  def __getitem__(self, _):
    # Break contract and return a random span.
    ix = torch.randint(self.dataset_size - self.block_size, (1,))
    x = torch.from_numpy(self.data[ix:ix+self.block_size].astype(np.int64))
    # pin array x, which allows us to move them to GPU asynchronously (non_blocking=True)
    x = x.pin_memory().to('cuda', non_blocking=True)
    return x
