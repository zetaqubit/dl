"""Tunes hyperparams using Optuna."""

from absl import flags
from absl import app
import gin
import optuna

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

OPTUNA_DB = 'sqlite:////media/14tb/ml/models/zetaqubit/dl/optuna/optuna.db'

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
  lr = trial.suggest_float('learning_rate', 1e-5, 1e-4)
  prepare_gin_for_study([
      f'learning_rate = {lr}',
  ], trial.number)

  metrics = train_lib.train()
  return metrics['eval/loss_valid']


def tune(_):
  study = optuna.create_study(
    study_name=f'{FLAGS.model_name}/{FLAGS.exp_name}',
    direction='minimize', storage=OPTUNA_DB)
  study.optimize(objective, n_trials=4)


if __name__ == '__main__':
  flags.mark_flags_as_required(['model_name', 'exp_name'])
  app.run(tune)
