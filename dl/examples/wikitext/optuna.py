"""Tunes hyperparams using Optuna."""

import os

from absl import flags
from absl import app
import gin

from dl.examples.wikitext import train_lib

flags.DEFINE_string('exp_name', None, 'Experiment name.')
flags.DEFINE_string('model_name', None, 'Name of the model - one of configs/.')
flags.DEFINE_multi_string(
    'ginc', [],
    'List of config files, relative to configs/.')
flags.DEFINE_multi_string(
    'ginp', [],
    'Newline separated list of Gin parameter bindings.')

FLAGS = flags.FLAGS


def prepare_gin_for_study():
  gin.clear_config(clear_constants=True)


def tune(_):
  ...


if __name__ == '__main__':
  app.run(tune)
  flags.mark_flags_as_required(['model_name', 'exp_name'])
