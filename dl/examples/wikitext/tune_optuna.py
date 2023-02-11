"""Tunes hyperparams using Optuna."""

import os

from absl import flags
from absl import app
import gin
import optuna
import tensorflow as tf

from dl.examples.wikitext import train_lib

flags.DEFINE_string('exp_name', None, 'Experiment name.')
flags.DEFINE_string('model_name', None, 'Name of the model - one of configs/.')
flags.DEFINE_multi_string(
    'ginc', [],
    'List of config files, relative to configs/.')
flags.DEFINE_multi_string(
    'ginp', [],
    'Newline separated list of Gin parameter bindings.')
flags.DEFINE_integer('n_trials', 10, 'Number of trials to run.')

FLAGS = flags.FLAGS

OPTUNA_DB = 'sqlite:////media/14tb/ml/models/zetaqubit/dl/optuna/optuna.db'
MODEL_DIR = '/media/14tb/ml/models/zetaqubit/dl/examples/wikitext'

def prepare_gin_for_study(gin_overrides, trial_number):
  configs = [f'dl/examples/wikitext/configs/{f}'
             for f in [FLAGS.model_name] + FLAGS.ginc]
  configs = [f'{f}.gin' if not f.endswith('.gin') else f for f in configs]
  gin_params = FLAGS.ginp + gin_overrides + [
      f'exp_name = "{FLAGS.exp_name}/{trial_number}"',
  ]
  print(configs, gin_params)

  gin.clear_config(clear_constants=True)
  gin.parse_config_files_and_bindings(configs, gin_params)


def objective(trial):
  lr = trial.suggest_float('learning_rate', 5e-6, 1e-3)
  prepare_gin_for_study([
      f'learning_rate = {lr}',
  ], trial.number)

  metrics = train_lib.train()

  dir = os.path.join(MODEL_DIR, FLAGS.model_name, FLAGS.exp_name,
                     str(trial.number))
  with tf.summary.create_file_writer(dir).as_default():
    step = metrics['step']
    for k, v in metrics.items():
      tf.summary.scalar(k, v, step=step)

  return metrics['eval/loss_valid']


def tune(_):
  study_name=f'{FLAGS.model_name}/{FLAGS.exp_name}'
  try:
    study = optuna.create_study(
      study_name=study_name, storage=OPTUNA_DB,
      direction='minimize')
  except optuna.exceptions.DuplicatedStudyError:
    yes_or_no = input(f'Study {study_name} already exists. Resume it [y/N]? ')
    if yes_or_no == 'y':
      study = optuna.load_study(study_name=study_name, storage=OPTUNA_DB)
    else:
      return

  tb_callback = optuna.integration.TensorBoardCallback(
    dirname=f'{MODEL_DIR}/{study_name}', metric_name='eval/loss_valid'
  )
  study.optimize(objective, n_trials=FLAGS.n_trials, callbacks=[tb_callback])


if __name__ == '__main__':
  flags.mark_flags_as_required(['model_name', 'exp_name'])
  app.run(tune)
