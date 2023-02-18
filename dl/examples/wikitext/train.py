"""Command-line for training a transformer on wikitext."""

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
flags.DEFINE_string(
    'resume', '',
    'Whether to resume training from a found checkpoint. Options are: \n'
    '  "": delete the directory and retrain from scratch (default)\n'
    '  "model": resume with model params from last checkpoint\n'
    '  "model_opt": resume with model param and optimizer.')

FLAGS = flags.FLAGS


def train(_):
  configs = [f'dl/examples/wikitext/configs/{f}'
             for f in [FLAGS.model_name] + FLAGS.ginc]
  configs = [f'{f}.gin' if not f.endswith('.gin') else f for f in configs]
  gin_params = FLAGS.ginp + [
      f'exp_name = "{FLAGS.exp_name}"',
      f'resume = "{FLAGS.resume}"',
  ]
  print(configs, gin_params)
  gin.parse_config_files_and_bindings(configs, gin_params)
  final_metrics = train_lib.train()
  print('final metrics:', final_metrics)


if __name__ == '__main__':
  flags.mark_flags_as_required(['model_name', 'exp_name'])
  app.run(train)
