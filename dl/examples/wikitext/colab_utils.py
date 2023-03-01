from importlib import reload

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

def create_gpt(model='small-model'):
  from dl.transformer import transformer
  clear()
  gin.parse_config_files_and_bindings(
      [f'dl/examples/wikitext/configs/{model}.gin'], [])
  gpt = transformer.GPT()
  return gpt

def create_rnn(model='small-model'):
  from dl.rnn import rnn
  clear()
  gin.parse_config_files_and_bindings([f'dl/rnn/configs/{model}.gin'], [])
  m = rnn.RNN()
  return m
