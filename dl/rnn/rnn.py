"""Implementations of various RNN architectures."""

import gin.torch
from einops import rearrange
import torch
import torch.nn.functional as F
from torch import nn, einsum

from dl.data import tokenizers

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

    hs_outs = [None] * len(self.rnn_cells)
    for i, cell in enumerate(self.rnn_cells):
      x, hs_outs[i] = cell(x, hs[:, i, :])
    return x, torch.stack(hs_outs, dim=1)


@gin.configurable
class RnnLM(nn.Module):
  """Language model backed by a RNN."""
  def __init__(self, n_layers: int, dim: int, max_seq_len: int, vocab: int):
    super().__init__()
    self.n_layers = n_layers
    self.dim = dim
    self.vocab = vocab
    self.rnn = RNN(n_layers)
    self.wte = nn.Embedding(vocab, dim)  # token embeddings
    self.lm_head = nn.Linear(dim, vocab, bias=False)
    self.lm_head.weight = self.wte.weight  # tie embedding weight

    self.max_seq_len = max_seq_len

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
    hs = torch.zeros((b, self.n_layers, self.dim), device=ids.device)
    logits_seq = [None] * seq_len
    x = xs[:, 0, :]
    for t in range(seq_len):
      x, hs = self.rnn(x, hs)
      logits = self.lm_head(x)
      logits_seq[t] = logits
      if not teacher_force_mask[t]:
        probs = F.softmax(logits, dim=-1)
        id = torch.multinomial(probs, 1)  # [b, 1]
        x = self.wte(id)  # [b, 1, dim]
        x = rearrange(x, 'b 1 d -> b d')

    return torch.stack(logits_seq, axis=1)

  def init_weights_(self, module):
    if isinstance(module, nn.Linear):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.2)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.2)

@gin.configurable
class GenerativeRnnModel(nn.Module):
  """Wrapper around a RNN model, providing additional capabilities.

  Capabilities provided:
    - Language modeling loss.
    - Tokenization (text str <-> ids).
    - Generate text via a prompt.
  """
  def __init__(self, net: nn.Module,
               tokenizer: tokenizers.Tokenizer,
               ignore_index: int):
    super().__init__()
    self.net = net
    self.tokenizer = tokenizer
    self.ignore_index = ignore_index
    self.max_seq_len = net.max_seq_len

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
               ):  # [batch, seq] -> [batch, seq_len]
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
    hs = torch.zeros((b, self.net.n_layers, self.net.dim), device=ids.device)
    for t in range(prompt_len - 1):
      _, hs = self.net.rnn(xs[:, t, :], hs)

    # Start generation.
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
    self.net.train(was_training)

    out = out[:, prompt_len:]
    out = self.tokenizer.decode_batch(out)
    if not was_batched: out = out[0]
    return out
