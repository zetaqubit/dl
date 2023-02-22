from importlib import reload

import numpy as np
import torch

def create_gpt(model='small-model'):
  import gin
  from dl.transformer import transformer
  gin.parse_config_files_and_bindings(
      [f'dl/examples/wikitext/configs/{model}.gin'], [])
  gpt = transformer.GPT()
  return gpt


def magic(cmd):
  get_ipython().magic(cmd)

def autoreload():
  magic('%load_ext autoreload')
  magic('%autoreload 2')

def save_history():
  magic('%history -f /tmp/ipython.py')
  print('Saved history to /tmp/ipython.py')
