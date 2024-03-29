"""Implementations of various RNN architectures."""
from typing import Callable

import gin.torch
from einops import rearrange
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, einsum

from dl.data import tokenizers

gin.external_configurable(torch.sigmoid, module='torch')
gin.external_configurable(torch.tanh, module='torch')


@gin.configurable
class RNNCell(nn.Module):
  def __init__(self, hidden_size, activation_hh):
    super().__init__()
    self.activation_hh = activation_hh
    self.wxh = nn.Linear(hidden_size, hidden_size)
    self.whh = nn.Linear(hidden_size, hidden_size)

  def forward(self, x, h):
    """
    Args:
      x: [batch, hidden_size]
      h: [batch, hidden_size]
    """
    h_t = self.activation_hh(self.whh(h) + self.wxh(x))
    return h_t


@gin.configurable
class GRUCell(nn.Module):
  def __init__(self, hidden_size):
    super().__init__()
    self.wz = nn.Linear(2 * hidden_size, hidden_size)
    self.wr = nn.Linear(2 * hidden_size, hidden_size)
    self.wh = nn.Linear(2 * hidden_size, hidden_size)

  def forward(self, x, h):
    """
    Args:
      x: [batch, hidden_size]
      h: [batch, hidden_size]
    """
    hx = torch.cat((h, x), axis=1)
    z = F.sigmoid(self.wz(hx))
    r = F.sigmoid(self.wr(hx))
    h_tilde = torch.cat((r * h, x), axis=1)
    h_tilde = F.tanh(self.wh(h_tilde))
    h_t = (1 - z) * h + z * h_tilde
    return h_t


@gin.configurable
class RNN(nn.Module):
  def __init__(self, n_layers, hidden_size, activation_ho, cell_fn):
    super().__init__()
    self.n_layers = n_layers
    self.hidden_size = hidden_size
    self.cells = nn.ModuleList([cell_fn(hidden_size=hidden_size)
                                for _ in range(n_layers)])
    self.who = nn.Linear(hidden_size, hidden_size)
    self.activation_ho = activation_ho

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

    hs_outs = [None] * len(self.cells)
    for i, cell in enumerate(self.cells):
      hs_outs[i] = cell(x, hs[:, i, :])
      x = hs_outs[i]
    out = self.who(hs_outs[-1])
    if self.activation_ho:
      out = self.activation_ho(out)
    return out, torch.stack(hs_outs, dim=1)

  def initial_state(self, x_shape):
    b = x_shape[0] if len(x_shape) > 1 else 1
    hs = torch.zeros((b, self.n_layers, self.hidden_size), device=self.device)
    return hs

  @property
  def device(self):
    return next(self.parameters()).device


@gin.configurable
class LSTMCell(nn.Module):
  def __init__(self, hidden_size, activation_g, activation_c,
               activation_h):
    super().__init__()
    self.forget_w = nn.Linear(2 * hidden_size, hidden_size)
    self.input_w = nn.Linear(2 * hidden_size, hidden_size)
    self.output_w = nn.Linear(2 * hidden_size, hidden_size)
    self.hidden_w = nn.Linear(2 * hidden_size, hidden_size)
    self.activation_g = activation_g
    self.activation_c = activation_c
    self.activation_h = activation_h

  def forward(self, x, h, c):
    """
    Args:
      x: [batch, hidden_size]
      h: [batch, hidden_size]
    """
    xh = torch.cat((x, h), axis=1)
    f_t = self.activation_g(self.forget_w(xh))
    i_t = self.activation_g(self.input_w(xh))
    o_t = self.activation_g(self.output_w(xh))
    c_t_tilde = self.activation_c(self.hidden_w(xh))
    c_t = f_t * c + i_t * c_t_tilde
    h_t = o_t * self.activation_h(c_t)
    return h_t, c_t


@gin.configurable
class LSTM(nn.Module):
  def __init__(self, n_layers, hidden_size, activation_ho):
    super().__init__()
    self.n_layers = n_layers
    self.hidden_size = hidden_size
    self.cells = nn.ModuleList([LSTMCell(hidden_size=hidden_size)
                                for _ in range(n_layers)])
    self.who = nn.Linear(hidden_size, hidden_size)
    self.activation_ho = activation_ho

  def forward(self, x, hiddens):
    """Runs RNN forward by 1 timestep.
    Args:
      x: [batch, hidden_size]
      hs: [batch, n_layers, hidden_size]
    Outputs:
      x: output of the top layer. [batch, hidden_size]
      hs_t: new hidden states. [batch, n_layers, hidden_size]
    """
    hs, cs = hiddens
    x_b, x_hidden = x.shape
    b, n, hidden_size = hs.shape
    assert b == x_b
    assert n == self.n_layers
    assert hidden_size == x_hidden
    assert hs.shape == cs.shape

    hs_outs = [None] * len(self.cells)
    cs_outs = [None] * len(self.cells)
    for i, cell in enumerate(self.cells):
      hs_outs[i], cs_outs[i] = cell(x, hs[:, i, :], cs[:, i, :])
      x = hs_outs[i]
    out = self.who(hs_outs[-1])
    if self.activation_ho:
      out = self.activation_ho(out)
    hs_t = torch.stack(hs_outs, dim=1)
    cs_t = torch.stack(cs_outs, dim=1)
    return out, (hs_t, cs_t)

  def initial_state(self, x_shape):
    b = x_shape[0] if len(x_shape) > 1 else 1
    hs = torch.zeros((b, self.n_layers, self.hidden_size), device=self.device)
    cs = torch.zeros((b, self.n_layers, self.hidden_size), device=self.device)
    return hs, cs

  @property
  def device(self):
    return next(self.parameters()).device


@gin.configurable
class RnnLM(nn.Module):
  """Language model backed by a RNN."""
  def __init__(self, n_layers: int, dim: int, vocab: int,
               rnn: Callable[[], nn.Module]):
    super().__init__()
    self.n_layers = n_layers
    self.dim = dim
    self.vocab = vocab
    self.rnn = rnn(n_layers=n_layers, hidden_size=dim)
    self.wte = nn.Embedding(vocab, dim)  # token embeddings
    self.lm_head = nn.Linear(dim, vocab, bias=False)
    self.lm_head.weight = self.wte.weight  # tie embedding weight

    # Initialize weights.
    self.apply(self.init_weights_)

  def forward(self, ids, teacher_force_mask=1):
    """
    input: ids [b, t]
    output: logits [b, t, vocab]
    """
    b, seq_len = ids.shape
    if not hasattr(teacher_force_mask, '__len__') or len(teacher_force_mask) == 1:
      teacher_force_mask = torch.full(
        (seq_len,), teacher_force_mask, device=ids.device)
    assert len(teacher_force_mask) == seq_len
    xs = self.wte(ids)  # [b, seq_len, dim]
    hs = self.rnn.initial_state(xs.shape)
    logits_seq = [None] * seq_len
    x = xs[:, 0, :]
    for t in range(seq_len):
      y, hs = self.rnn(x, hs)
      logits = self.lm_head(y)
      logits_seq[t] = logits
      if t == seq_len - 1:
        break
      if teacher_force_mask[t+1]:
        x = xs[:, t+1, :]
      else:
        probs = F.softmax(logits, dim=-1)
        id = torch.multinomial(probs, 1)  # [b, 1]
        x = self.wte(id)  # [b, 1, dim]
        x = rearrange(x, 'b 1 d -> b d')

    return torch.stack(logits_seq, axis=1)

  def init_weights_(self, module):
    if isinstance(module, nn.Linear):
        torch.nn.init.normal_(module.weight, mean=0.0, std=np.sqrt(1/self.dim))
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=np.sqrt(1/self.dim))

@gin.configurable
class GenerativeRnnModel(nn.Module):
  """Wrapper around a RNN model, providing additional capabilities.

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

  def forward(self, x, teacher_forcing='all'):  # [batch, seq] -> loss
    inputs, targets = x[:, :-1], x[:, 1:]
    b, t = inputs.shape

    if teacher_forcing == 'all':
      mask = 1
    elif teacher_forcing == 'none':
      mask = 0
    elif teacher_forcing == 'first_half':
      mask = torch.zeros(t, device=inputs.device)
      mask[:t//2] = 1
    else:
      raise ValueError(f'Unknown teacher forcing opt: {teacher_forcing}')

    logits = self.net(inputs, mask)
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
    b, prompt_len = ids.shape

    # Run RNN through the prompt.
    xs = self.net.wte(ids)  # [b, prompt_len, dim]
    hs = self.net.rnn.initial_state(xs.shape)
    for t in range(prompt_len - 1):
      _, hs = self.net.rnn(xs[:, t, :], hs)

    # Start generation.
    while True:
      out = ids
      x = xs[:, -1, :]
      for _ in range(seq_len):
        y, hs = self.net.rnn(x, hs)
        logits = self.net.lm_head(y)  # [b, v]
        probs = F.softmax(logits / temperature, dim=-1)
        sample = torch.multinomial(probs, 1)  # [b, 1]
        out = torch.cat((out, sample), dim=-1)  # [b, s]
        x = self.net.wte(sample)
        x = rearrange(x, 'b 1 d -> b d')  # t = 1

      out = out[:, prompt_len:]
      out = self.tokenizer.decode_batch(out)
      if not was_batched: out = out[0]
      yield out
      if not continuous: break
    self.net.train(was_training)

  # Same as above, but non-generator.
  def generate_text(self, **kwargs):
    was_training = self.net.training
    self.net.eval()
    out = next(self.generate(**kwargs))
    self.net.train(was_training)
    return out
