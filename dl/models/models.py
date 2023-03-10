"""Factory for creating models."""

import gin
from torch import nn

from dl.transformer import transformer
from dl.rnn import rnn

@gin.configurable(module='models')
def create(net: nn.Module):
  return net
