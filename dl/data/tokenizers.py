"""Selection of tokenizers for converting between text and ids."""

import abc
from typing import List
import functools

import gin
import numpy as np
import tiktoken
import transformers as hf_transformers


class Tokenizer(abc.ABC):
  @abc.abstractmethod
  def encode_batch(self, texts: List[str]) -> List[List[int]]:
    pass

  @abc.abstractmethod
  def decode_batch(self, ids: List[List[int]]) -> List[int]:
    pass

  @abc.abstractproperty
  def padding_id() -> int:
    pass

  def encode(self, text: str) -> List[int]:
    return self.encode_batch([text])[0]

  def decode(self, ids: List[int]) -> str:
    return self.decode_batch([ids])[0]


@gin.configurable(module='tokenizers')
def create(library: str, tok_type: str,
           max_seq_len=None) -> Tokenizer:
  tok = None
  if library == 'tiktoken':
    tok = TikTokenTokenizer(tok_type)
  elif library == 'huggingface':
    tok = HuggingFaceTokenizer(tok_type)
  elif tok_type == 'char':
    tok = CharTokenizer()

  if not tok:
    raise ValueError(f'Unknown tokenizer lib: {library} and type: {tok_type}.')

  if max_seq_len:
    tok = PaddingTokenizer(tok, max_seq_len)

  return tok


class TikTokenTokenizer(Tokenizer):
  def __init__(self, tok_type: str):
    self.tokenizer = tiktoken.get_encoding(tok_type)

  def encode_batch(self, texts, **kwargs):
    return self.tokenizer.encode_batch(texts, **kwargs)

  def decode_batch(self, ids, **kwargs):
    return np.array([self.tokenizer.decode(seq, **kwargs) for seq in ids])

  @property
  def padding_id(self) -> int:
    raise NotImplementedError


class HuggingFaceTokenizer(Tokenizer):
  def __init__(self, tok_type: str):
    self.tokenizer = hf_transformers.AutoTokenizer.from_pretrained(tok_type)
    self.tokenizer.pad_token = self.tokenizer.eos_token

  def encode_batch(self, texts, **kwargs):
    return self.tokenizer.batch_encode_plus(texts, **kwargs).input_ids

  def decode_batch(self, ids, **kwargs):
    return self.tokenizer.batch_decode(ids, **kwargs)

  @property
  def padding_id(self) -> int:
    return self.tokenizer.pad_token_id


class CharTokenizer(Tokenizer):
  def __init__(self, drop_non_ascii=True):
    self.drop_non_ascii = drop_non_ascii

  def encode_batch(self, texts):
    encode_fn = lambda s: s.encode(
      'ascii', 'ignore' if self.drop_non_ascii else 'strict')
    return [list(encode_fn(text)) for text in texts]

  def decode_batch(self, ids):
    return [''.join(chr(id) for seq in ids for id in seq)]

  @property
  def padding_id(self) -> int:
    return 0


class PaddingTokenizer(Tokenizer):
  def __init__(self, tokenizer: Tokenizer, max_seq_len: int):
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
