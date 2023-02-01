"""Implementation of the transformer from Attention is All You Need."""

from einops import rearrange
import torch
import torch.nn.functional as F
from torch import nn, einsum


class SelfAttention(nn.Module):
  def __init__(self, dim, causal=False):
    super().__init__()
    self.scaling_factor = dim ** -0.5
    self.causal = causal
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

    dots = einsum('b i d, b j d -> b i j', q, k) * self.scaling_factor
    if self.causal:
      j = dots.shape[-1]
      mask = torch.ones((j, j), dtype=bool).triu(diagonal=1)
      dots.masked_fill_(mask, float('-inf'))
    attention = F.softmax(dots, dim=-1)  # b i j
    summed = einsum('b i j, b j d -> b i d', attention, v)
    return summed


class TransformerBlock(nn.Module):
  def __init__(self, dim: int, causal: bool):
    super().__init__()
    self.ln_1 = nn.LayerNorm(dim)
    self.attention = SelfAttention(dim, causal)
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


class Decoder(nn.Module):
  """Decoder stack with causal attention.

  Note: unlike in Vaswani et al, this decoder does not attend to encoder stack.
  """
  def __init__(self, n_layers, dim):
    super().__init__()
    self.decoder_stack = nn.Sequential(
      *[TransformerBlock(dim, causal=True) for _ in range(n_layers)]
    )

  def forward(self, x):
    return self.decoder_stack(x)


class AbsolutePositionalEmbedding(nn.Module):
  def __init__(self, dim: int, max_seq_len: int):
    super().__init__()
    self.max_seq_len = max_seq_len
    self.emb = nn.Embedding(max_seq_len, dim)
    self.scale = dim ** -0.5

  def forward(self, x):  # [b, s]  ->  [b, s, e]
    seq_len = x.shape[1]
    assert seq_len <= self.max_seq_len, "input length > max_seq_len"
    pos = torch.arange(seq_len)
    pos_emb = self.emb(pos) * self.scale
    return pos_emb


class GPT(nn.Module):
  def __init__(self, n_layers: int, dim: int, max_seq_len: int, vocab: int):
    super().__init__()
    self.max_seq_len = max_seq_len
    self.wte = nn.Embedding(vocab, dim)  # token embeddings
    self.wpe = AbsolutePositionalEmbedding(dim, max_seq_len)
    self.transformers = Decoder(n_layers, dim)
    self.lm_head = nn.Linear(dim, vocab, bias=False)
    self.lm_head.weight = self.wte.weight  # tie embedding weight

  def forward(self, ids):
    x = self.wte(ids) + self.wpe(ids)
    x = self.transformers(x)
    x = self.lm_head(x)
    return x


class AutoregressiveModel(nn.Module):
  def __init__(self, net: nn.Module, ignore_index=-100):
    super().__init__()
    self.net = net
    self.ignore_index = ignore_index
    self.max_seq_len = net.max_seq_len

  def forward(self, x):  # [batch, seq] -> loss
    inputs, targets = x[:, :-1], x[:, 1:]
    logits = self.net(inputs)
    logits = rearrange(logits, 'b s v -> (b s) v')
    targets = rearrange(targets, 'b s -> (b s)')
    loss = F.cross_entropy(logits, targets, ignore_index=self.ignore_index)
    return loss

  def generate(self,
               prompt,
               seq_len,
               temperature=1,
               ):  # [batch, seq] -> [batch, seq_len]
    was_training = self.net.training
    self.net.eval()

    b, t = prompt.shape
    out = prompt
    for _ in range(seq_len):
      x = out[:, -self.max_seq_len:]
      logits = self.net(x)[:, -1]
      probs = F.softmax(logits / temperature, dim=-1)
      sample = torch.multinomial(probs, 1)
      out = torch.cat((out, sample), dim=-1)
      # todo: handle eos and break

    out = out[:, t:]
    self.net.train(was_training)
    return out
