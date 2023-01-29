"""Implementation of the transformer from Attention is All You Need."""

import math

import torch
import torch.nn.functional as F
from torch import nn, einsum


class SelfAttention(nn.Module):
  def __init__(self, dim):
    super().__init__()
    self.scaling_factor = 1 / math.sqrt(dim)
    self.to_q = nn.Linear(dim, dim)
    self.to_k = nn.Linear(dim, dim)
    self.to_v = nn.Linear(dim, dim)

  def forward(self, x):
    """
    Args:
      x: [batch, seq, dim]
    """
    q = self.to_q(x)
    k = self.to_k(x)
    v = self.to_v(x)

    dots = einsum('b i d, b j d -> b i j', q, k)  # b i j
    scaled = dots * self.scaling_factor
    attention = F.softmax(scaled, dim=-1)  # b i j
    summed = einsum('b i j, b j d -> b i d', attention, v)
    return summed


class TransformerBlock(nn.Module):
  def __init__(self, dim):
    super().__init__()
    self.ln_1 = nn.LayerNorm(dim)
    self.attention = SelfAttention(dim)
    self.ln_2 = nn.LayerNorm(dim)
    self.ff = nn.Linear(dim, dim)

  def forward(self, x):
    x = x + self.attention(self.ln_1(x))
    x = x + self.ff(self.ln_2(x))
    return x


class Encoder(nn.Module):
  def __init__(self, n_layers, dim):
    super().__init__()
    self.encoder_stack = nn.Sequential(
      *[TransformerBlock(dim) for _ in range(n_layers)]
    )

  def forward(self, x):
    return self.encoder_stack(x)
