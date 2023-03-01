"""Implementations of various RNN architectures."""

import gin.torch
import torch
from torch import nn, einsum

gin.external_configurable(torch.sigmoid, module='torch')
gin.external_configurable(torch.tanh, module='torch')


@gin.configurable
class RNNCell(nn.Module):
  def __init__(self, hidden_size, activation_hh, activation_hx):
    super().__init__()
    self.activation_hh = activation_hh
    self.activation_hx = activation_hx
    self.wxh = nn.Linear(hidden_size, hidden_size)
    self.whh = nn.Linear(hidden_size, hidden_size)
    self.who = nn.Linear(hidden_size, hidden_size)

  def forward(self, x, h):
    """
    Args:
      x: [batch, hidden_size]
      h: [batch, hidden_size]
    """
    h_t = self.activation_hh(self.whh(h) + self.wxh(x))
    x_t = self.activation_hx(self.who(h_t))
    return x_t, h_t


@gin.configurable
class RNN(nn.Module):
  def __init__(self, n_layers):
    super().__init__()
    self.n_layers = n_layers
    self.rnn_cells = nn.ModuleList([RNNCell() for _ in range(n_layers)])

  def forward(self, x, hs):
    """Runs RNN forward by 1 timestep.
    Args:
      x: [batch, hidden_size]
      hs: [batch, n_layers, hidden_size]
    Outputs:
      x: output of the top layer. [batch, hidden_size]
      hs_t: new hidden states. [batch, n_layers, hidden_size]
    """
    x_b, x_hidden = x.shape
    b, n, hidden_size = hs.shape
    assert b == x_b
    assert n == self.n_layers
    assert hidden_size == x_hidden

    hs_t = torch.empty_like(hs)
    for i, cell in enumerate(self.rnn_cells):
      x, hs_t[:, i, :] = cell(x, hs[:, i, :])
    return x, hs_t
