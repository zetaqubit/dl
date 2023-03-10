"""Selection of tokenizers for converting between text and ids."""

import abc
from typing import List
import functools

import gin
import numpy as np
import tiktoken
import transformers as hf_transformers


class Tokenizer(abc.ABC):
  # Capture constructor args to make this class pickle-able.
  def __init__(self, **kwargs):
    self.args = frozenset(kwargs.items())

  def __getstate__(self):
    return self.args

  def __setstate__(self, state):
    state = dict(state)
    self.__init__(**state)

  @abc.abstractmethod
  def encode_batch(self, texts: List[str]) -> List[List[int]]:
    pass

  @abc.abstractmethod
  def decode_batch(self, ids: List[List[int]]) -> List[int]:
    pass

  @abc.abstractproperty
  def padding_id() -> int:
    pass

  @abc.abstractproperty
  def vocab_size() -> int:
    pass

  def encode(self, text: str) -> List[int]:
    return self.encode_batch([text])[0]

  def decode(self, ids: List[int]) -> str:
    return self.decode_batch([ids])[0]


@gin.configurable(module='tokenizers')
def create(tok_type: str, max_seq_len=None) -> Tokenizer:
  tok = None
  if tok_type in ('char',):
    tok = CharTokenizer()
  elif tok_type in ('gpt2',):
    tok = TikTokenTokenizer(tok_type)
  else:
    tok = HuggingFaceTokenizer(tok_type)

  if not tok:
    raise ValueError(f'Unknown tokenizer type: {tok_type}.')

  if max_seq_len:
    tok = PaddingTokenizer(tok, max_seq_len)

  return tok


class TikTokenTokenizer(Tokenizer):
  def __init__(self, tok_type: str):
    super().__init__(tok_type=tok_type)
    self.tokenizer = tiktoken.get_encoding(tok_type)

  def encode_batch(self, texts, **kwargs):
    return self.tokenizer.encode_batch(texts, allowed_special='all', **kwargs)

  def decode_batch(self, ids, **kwargs):
    return np.array([self.tokenizer.decode(list(seq), **kwargs) for seq in ids])

  @property
  def padding_id(self) -> int:
    return self.tokenizer.eot_token

  @property
  def vocab_size(self) -> int:
    return self.tokenizer.max_token_value + 1


class HuggingFaceTokenizer(Tokenizer):
  def __init__(self, tok_type: str):
    super().__init__(tok_type=tok_type)
    self.tokenizer = hf_transformers.AutoTokenizer.from_pretrained(tok_type)
    self.tokenizer.pad_token = self.tokenizer.eos_token

  def encode_batch(self, texts, **kwargs):
    return self.tokenizer.batch_encode_plus(texts, **kwargs).input_ids

  def decode_batch(self, ids, **kwargs):
    return self.tokenizer.batch_decode(ids, **kwargs)

  @property
  def padding_id(self) -> int:
    return self.tokenizer.pad_token_id

  @property
  def vocab_size(self) -> int:
    return len(self.tokenizer.get_vocab())


class CharTokenizer(Tokenizer):
  def __init__(self, drop_non_ascii=True):
    super().__init__(drop_non_ascii=drop_non_ascii)
    self.drop_non_ascii = drop_non_ascii

  def encode_batch(self, texts):
    encode_fn = lambda s: s.encode(
      'ascii', 'ignore' if self.drop_non_ascii else 'strict')
    return [list(encode_fn(text)) for text in texts]

  def decode_batch(self, ids):
    return [''.join(chr(id) for id in seq) for seq in ids]

  @property
  def padding_id(self) -> int:
    return 0

  @property
  def vocab_size(self) -> int:
    return 128


class PaddingTokenizer(Tokenizer):
  def __init__(self, tokenizer: Tokenizer, max_seq_len: int):
    super().__init__(tokenizer=tokenizer, max_seq_len=max_seq_len)
    self.tokenizer = tokenizer
    self.max_seq_len = max_seq_len

  def encode_batch(self, texts, **kwargs):
    ids = self.tokenizer.encode_batch(texts, **kwargs)
    for seq in ids:
      to_pad = self.max_seq_len - len(seq)
      if to_pad <= 0:
        del seq[self.max_seq_len:]
      else:
        seq.extend([self.tokenizer.padding_id] * to_pad)
    return ids

  def decode_batch(self, ids, **kwargs):
    return self.tokenizer.decode_batch(ids, **kwargs)

  @property
  def padding_id(self) -> int:
    return self.tokenizer.padding_id

  @property
  def vocab_size(self) -> int:
    return self.tokenizer.vocab_size
