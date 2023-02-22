"""Implementation of the transformer from Attention is All You Need."""

import gin
from einops import rearrange
from transformers import AutoTokenizer
import torch
import torch.nn.functional as F
from torch import nn, einsum


@gin.configurable
class SelfAttention(nn.Module):
  def __init__(self, dim: int, heads: int, causal=False):
    super().__init__()
    self.scaling_factor = dim ** -0.5
    self.heads = heads
    self.causal = causal
    self.to_q = nn.Linear(dim, dim)
    self.to_k = nn.Linear(dim, dim)
    self.to_v = nn.Linear(dim, dim)

  def forward(self, x):
    """
    Args:
      x: [batch, seq, dim]
    """
    device = x.device
    q = self.to_q(x)
    k = self.to_k(x)
    v = self.to_v(x)
    q = rearrange(q, 'b s (h d) -> b h s d', h=self.heads)
    k = rearrange(k, 'b s (h d) -> b h s d', h=self.heads)
    v = rearrange(v, 'b s (h d) -> b h s d', h=self.heads)

    dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scaling_factor
    if self.causal:
      j = dots.shape[-1]
      mask = torch.ones((j, j), dtype=bool, device=device).triu(diagonal=1)
      dots.masked_fill_(mask, float('-inf'))
    attention = F.softmax(dots, dim=-1)  # b h i j
    summed = einsum('b h i j, b h j d -> b h i d', attention, v)
    concated = rearrange(summed, 'b h i d -> b i (h d)')
    return concated


@gin.configurable
class TransformerBlock(nn.Module):
  def __init__(self, dim: int, causal: bool):
    super().__init__()
    dim_internal = dim * 4
    self.ln_1 = nn.LayerNorm(dim)
    self.attention = SelfAttention(dim=dim, causal=causal)
    self.ln_2 = nn.LayerNorm(dim)
    self.ff = nn.Sequential(
      nn.Linear(dim, dim_internal),
      nn.GELU(),
      nn.Linear(dim_internal, dim)
    )

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
class AbsolutePositionalEmbedding(nn.Module):
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
class GPT(nn.Module):
  def __init__(self, n_layers: int, dim: int, max_seq_len: int, vocab: int):
    super().__init__()
    self.max_seq_len = max_seq_len
    self.vocab = vocab
    self.wte = nn.Embedding(vocab, dim)  # token embeddings
    self.wpe = AbsolutePositionalEmbedding(dim, max_seq_len)
    self.transformers = Decoder(n_layers, dim)
    self.lm_head = nn.Linear(dim, vocab, bias=False)
    self.lm_head.weight = self.wte.weight  # tie embedding weight

    # Initialize weights.
    self.apply(self.init_weights_)
    # apply special scaled init to the residual projections, per GPT-2 paper
    for pn, p in self.named_parameters():
      if pn.endswith('c_proj.weight'):
        torch.nn.init.normal_(p, mean=0.0, std=0.02/torch.sqrt(2 * n_layers))

  def forward(self, ids):
    x = self.wte(ids) + self.wpe(ids)
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
  def __init__(self, net: nn.Module,
               tokenizer: AutoTokenizer,
               ignore_index: int):
    super().__init__()
    self.net = net
    self.tokenizer = tokenizer
    self.ignore_index = ignore_index
    self.max_seq_len = net.max_seq_len

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
               eos_token=50257,
               ):  # [batch, seq] -> [batch, seq_len]
    """Text prompt -> text output."""
    was_training = self.net.training
    self.net.eval()

    was_batched = True
    if isinstance(prompt, str):
      was_batched = False
      prompt = [prompt]
    ids = torch.tensor(self.tokenizer(prompt).input_ids, device='cuda')
    b, t = ids.shape
    out = ids
    for _ in range(seq_len):
      x = out[:, -self.max_seq_len:]  # [b, msl]
      logits = self.net(x)[:, -1]  # [b, v]
      probs = F.softmax(logits / temperature, dim=-1)
      sample = torch.multinomial(probs, 1)  # [b, 1]
      out = torch.cat((out, sample), dim=-1)  # [b, s]
      # todo: handle eos and break
      if (sample == eos_token).all():
        break
    self.net.train(was_training)

    out = out[:, t:]
    out = self.tokenizer.batch_decode(out)
    if not was_batched: out = out[0]

    return out
