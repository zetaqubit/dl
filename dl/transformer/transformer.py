"""Implementation of the transformer from Attention is All You Need."""
import math
from typing import Callable

from collections import OrderedDict
from einops import rearrange, repeat
import gin
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, einsum

from dl.data import tokenizers


@gin.configurable
class Attention(nn.Module):
  def __init__(self, dim: int, heads: int, causal=False, attn_bias=None):
    super().__init__()
    self.scaling_factor = dim ** -0.5
    self.heads = heads
    self.causal = causal
    self.attn_bias = attn_bias
    self.to_q = nn.Linear(dim, dim)
    self.to_k = nn.Linear(dim, dim)
    self.to_v = nn.Linear(dim, dim)
    self.to_out = nn.Linear(dim, dim)  # to_out for special init

  def forward(self, x, y=None):
    """
    Args:
      x: Main sequence used to compute the query. Also defines the output len.
         [batch, seq, dim]
      y: Optional cross-attention sequence to compute the key and values.
         If not supplied, falls back to x, becoming self-attention.
         [batch, seq, dim]
    """
    device = x.device
    if y is None: y = x
    q = self.to_q(x)
    k = self.to_k(y)
    v = self.to_v(y)
    q = rearrange(q, 'b i (h d) -> b h i d', h=self.heads)
    k = rearrange(k, 'b j (h d) -> b h j d', h=self.heads)
    v = rearrange(v, 'b j (h d) -> b h j d', h=self.heads)

    dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scaling_factor
    if self.attn_bias is not None:
      dots = dots + self.attn_bias(x)
    if self.causal:
      j = dots.shape[-1]
      mask = torch.ones((j, j), dtype=bool, device=device).triu(diagonal=1)
      dots.masked_fill_(mask, float('-inf'))
    attention = F.softmax(dots, dim=-1)  # b h i j
    summed = einsum('b h i j, b h j d -> b h i d', attention, v)
    concated = rearrange(summed, 'b h i d -> b i (h d)')
    out = self.to_out(concated)
    return out


@gin.configurable
class TransformerBlock(nn.Module):
  def __init__(self, dim: int, causal: bool):
    super().__init__()
    dim_internal = dim * 4
    self.ln_1 = nn.LayerNorm(dim)
    self.attention = Attention(dim=dim, causal=causal)
    self.ln_2 = nn.LayerNorm(dim)
    self.ff = nn.Sequential(OrderedDict([
      ('ff1', nn.Linear(dim, dim_internal)),
      ('gelu', nn.GELU()),
      ('to_out', nn.Linear(dim_internal, dim)),  # to_out for special init
    ]))

  def forward(self, x):
    x = x + self.attention(self.ln_1(x))
    x = x + self.ff(self.ln_2(x))
    return x


@gin.configurable
class Encoder(nn.Module):
  def __init__(self, n_layers, dim):
    super().__init__()
    self.encoder_stack = nn.Sequential(
      *[TransformerBlock(dim) for _ in range(n_layers)]
    )

  def forward(self, x):
    return self.encoder_stack(x)


@gin.configurable
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


@gin.configurable
class AbsolutePositionEmbedding(nn.Module):
  def __init__(self, dim: int, max_seq_len: int):
    super().__init__()
    self.max_seq_len = max_seq_len
    self.emb = nn.Embedding(max_seq_len, dim)
    self.scale = dim ** -0.5

  def forward(self, x):  # [b, s]  ->  [b, s, e]
    seq_len, device = x.shape[1], x.device
    assert seq_len <= self.max_seq_len, "input length > max_seq_len"
    pos = torch.arange(seq_len, device=device)
    pos_emb = self.emb(pos) * self.scale
    return pos_emb


@gin.configurable
class SinusoidalPositionEmbedding(nn.Module):
  def __init__(self, dim: int, max_seq_len: int):
    super().__init__()
    self.max_seq_len = max_seq_len
    pos = torch.arange(0, max_seq_len)[:, None]
    div = torch.arange(0, dim, 2) * math.log(10000) / dim
    div = torch.exp(-div)
    self.register_buffer('emb', torch.zeros((max_seq_len, dim)))
    self.emb[: , 0::2] = torch.sin(pos * div)
    self.emb[: , 1::2] = torch.cos(pos * div)

  def forward(self, x):  # [b, s]  ->  [b, s, e]
    seq_len = x.shape[1]
    assert seq_len <= self.max_seq_len, "input length > max_seq_len"
    pos_emb = self.emb[:seq_len, :]
    return pos_emb


@gin.configurable
class RelativePositionBias(nn.Module):
  def __init__(self, max_rel_pos: int):
    super().__init__()
    self.max_rel_pos = max_rel_pos
    self.bias = nn.Parameter(torch.rand(2 * max_rel_pos + 1))

  def forward(self, x):  # [b, s]  ->  [s, s, e]
    seq_len, device = x.shape[1], x.device
    pos = torch.arange(seq_len, device=device)  # [s]
    rel_pos = pos[None, :] - pos[:, None]  # [s, s]. Represents pos_q - pos_k.
    rel_pos.clamp_(-self.max_rel_pos, self.max_rel_pos)
    rel_pos += self.max_rel_pos  # convert to range [0, 2 * max_rel_pos]
    pos_bias = self.bias[rel_pos]  # [s, s]
    return pos_bias


# Rotary position embeddings, implemented efficiently as described in
# https://arxiv.org/pdf/2104.09864.pdf, section 3.4.2.
@gin.configurable
class RotaryPositionEmbedding(nn.Module):
  def __init__(self, max_seq_len: int, dim: int):
    super().__init__()
    self.max_seq_len = max_seq_len
    inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))  # [d]
    position = torch.arange(0, max_seq_len, dtype=torch.float)  # [s]
    sinusoid = torch.einsum("s, d -> s d", position, inv_freq)  # [s, d]
    sinusoid = repeat(sinusoid, 's d -> s (d j)', j=2)
    self.cos_mult = sinusoid.cos()  # [s, d]
    self.sin_mult = sinusoid.sin()  # [s, d]

  def rotate_pairs(self, x):
    x = rearrange(x, '... (d j) -> ... j d', j=2)
    x1, x2 = x[..., 0, :], x[..., 1, :]
    x = torch.stack((-x2, x1), dim=-2)
    x = rearrange(x, '... j d -> ... (d j)', j=2)
    return x

  def forward(self, x):  # [..., s, d] -> [..., s, d]
    _, seq, dim = x.shape
    assert dim % 2 == 0
    assert seq <= self.max_seq_len

    x_rot = self.rotate_pairs(x)
    rot = self.cos_mult * x + self.sin_mult * x_rot
    return rot


@gin.configurable
class GPT(nn.Module):
  def __init__(self, n_layers: int, dim: int, max_seq_len: int, vocab: int,
               pos_emb_fn: Callable[[], nn.Module]):
    super().__init__()
    self.max_seq_len = max_seq_len
    self.vocab = vocab
    self.wte = nn.Embedding(vocab, dim)  # token embeddings
    self.wpe = pos_emb_fn(dim=dim) if pos_emb_fn is not None else None
    self.transformers = Decoder(n_layers, dim)
    self.lm_head = nn.Linear(dim, vocab, bias=False)
    self.lm_head.weight = self.wte.weight  # tie embedding weight

    # Initialize weights.
    self.apply(self.init_weights_)
    # apply special scaled init to the residual projections, per GPT-2 paper
    for pn, p in self.named_parameters():
      if pn.endswith('to_out.weight'):
        torch.nn.init.normal_(p, mean=0.0, std=0.02/np.sqrt(2 * n_layers))

  def forward(self, ids):
    x = self.wte(ids)
    if self.wpe is not None:
      x = x + self.wpe(ids)
    x = self.transformers(x)
    x = self.lm_head(x)
    return x

  def init_weights_(self, module):
    if isinstance(module, nn.Linear):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


@gin.configurable
class AutoregressiveModel(nn.Module):
  """Wrapper around a transformer model, providing additional capabilities.

  Capabilities provided:
    - Language modeling loss.
    - Tokenization (text str <-> ids).
    - Generate text via a prompt.
  """
  def __init__(self, net: Callable[[], nn.Module]):
    super().__init__()
    self.tokenizer = tokenizers.create(max_seq_len=None)
    self.net = net(vocab=self.tokenizer.vocab_size)
    self.ignore_index = self.tokenizer.padding_id
    self.max_seq_len = self.net.max_seq_len

  def forward(self, x):  # [batch, seq] -> loss
    inputs, targets = x[:, :-1], x[:, 1:]
    logits = self.net(inputs)
    logits = rearrange(logits, 'b s v -> (b s) v')
    targets = rearrange(targets, 'b s -> (b s)')
    loss = F.cross_entropy(logits, targets, ignore_index=self.ignore_index)
    return loss

  @gin.configurable
  def generate(self,
               prompt,
               seq_len,
               temperature=1,
               continuous=True,
               ):  # [batch, seq] -> generator of [batch, seq_len]s
    """Text prompt -> text output."""
    was_training = self.net.training
    self.net.eval()

    was_batched = True
    if isinstance(prompt, str):
      was_batched = False
      prompt = [prompt]
    ids = torch.tensor(self.tokenizer.encode_batch(prompt), device='cuda')
    b, t = ids.shape

    # Start generation.
    while True:
      out = ids
      for _ in range(seq_len):
        x = out[:, -self.max_seq_len:]  # [b, msl]
        logits = self.net(x)[:, -1]  # [b, v]
        probs = F.softmax(logits / temperature, dim=-1)
        sample = torch.multinomial(probs, 1)  # [b, 1]
        out = torch.cat((out, sample), dim=-1)  # [b, s]

      ids_out = out[:, t:]
      text_out = self.tokenizer.decode_batch(ids_out)
      if not was_batched: text_out = text_out[0]
      yield text_out
      ids = ids_out
      if not continuous: break
    self.net.train(was_training)

  # Same as above, but non-generator.
  def generate_text(self, **kwargs):
    was_training = self.net.training
    self.net.eval()
    out = next(self.generate(**kwargs))
    self.net.train(was_training)
    return out
