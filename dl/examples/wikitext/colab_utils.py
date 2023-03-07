from importlib import reload
from IPython import get_ipython

import gin.torch
import numpy as np
import torch

gin.enter_interactive_mode()

def magic(cmd):
  get_ipython().magic(cmd)

def autoreload():
  magic('%load_ext autoreload')
  magic('%autoreload 2')


def save_history():
  magic('%history -f /tmp/ipython.py')
  print('Saved history to /tmp/ipython.py')

def clear():
  gin.clear_config(clear_constants=True)

def print_weights(net, names=None):
  for name, p in net.named_parameters():
    if names and name not in names: continue
    print(name, p)

def print_grads(net, names=None):
  for name, p in net.named_parameters():
    if names and name not in names: continue
    print(name, p.grad)

def print_weights_l2(net, names=None):
  for name, p in net.named_parameters():
    if names and name not in names: continue
    print(name, (p ** 2).sum())

def print_grads_l2(net, names=None):
  for name, p in net.named_parameters():
    if names and name not in names: continue
    print(name, (p.grad ** 2).sum())

def gradient_step(net, lr=1e-4):
  for p in net.parameters():
    p.data.add_(p.grad.data, alpha=-lr)

def create_gpt(model='small-model'):
  clear()
  from dl.transformer import transformer
  gin.parse_config_files_and_bindings(
      [f'dl/examples/wikitext/configs/{model}.gin'], [])
  gpt = transformer.GPT()
  return gpt

def create_rnn(model='rnn-debug'):
  clear()
  from dl.rnn import rnn
  from dl.transformer import transformer
  gin.parse_config_files_and_bindings([f'dl/rnn/configs/{model}.gin'], [])
  m = rnn.RnnLM()
  return m

def create_am(model='rnn-debug'):
  clear()
  from dl.rnn import rnn
  from dl.transformer import transformer
  from dl.data import tokenizers
  gin.parse_config_files_and_bindings([f'dl/rnn/configs/{model}.gin'], [])
  tok = tokenizers.create('char', max_seq_len=8)
  am = rnn.GenerativeRnnModel(tokenizer=tok)
  return am


autoreload()