"""Generates text based on a prompt."""
import os

from absl import flags
from absl import app
import gin
import torch
import transformers as hf_transformers

from dl.examples.wikitext import checkpoint
from dl.data import dataset
from dl.models import models

flags.DEFINE_string('model_name', None, 'Model name - one of configs.')
flags.DEFINE_string('exp_name', None, 'Experiment name to load from.')
flags.DEFINE_float('temperature', 1, 'Temperature to use during sampling.')
flags.DEFINE_integer('seq_len', None,
                     'Number of tokens to generate. Special cases: \n'
                     '  None: use default from the model.\n'
                     '  0: no limit - generate infinite tokens.\n')

FLAGS = flags.FLAGS


MODEL_DIR = '/media/14tb/ml/models/zetaqubit/dl/examples/wikitext'


def load_model(dir):
  model = models.create()
  checkpoint.load_ckpt(dir, model)
  model.eval()
  model.cuda()
  return model


def generate(_):
  exp_dir = os.path.join(MODEL_DIR, FLAGS.model_name, FLAGS.exp_name)
  gin.parse_config_file(f'{exp_dir}/config.gin')

  model = load_model(exp_dir)
  seq_len = FLAGS.seq_len
  if seq_len is None:
    seq_len = gin.query_parameter('%max_seq_len')
  if seq_len == 0:
    continuous = True
    seq_len = gin.query_parameter('%max_seq_len')
  else:
    continuous = False

  while True:
    try:
      prompt = input(f'Prompt: ')
    except (EOFError, KeyboardInterrupt):
      print()
      break
    try:
      for out_text in model.generate(
          prompt, seq_len=seq_len, temperature=FLAGS.temperature,
          continuous=continuous):
        print(out_text, end='')
      print()
    except (EOFError, KeyboardInterrupt):
      print()
      break

if __name__ == '__main__':
  app.run(generate)
  flags.mark_flags_as_required(['model_name', 'exp_name'])
