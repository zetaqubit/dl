"""Generates text based on a prompt."""
import os

from absl import flags
from absl import app
import gin
import torch
import transformers as hf_transformers

from dl.examples.wikitext import checkpoint
from dl.transformer import transformer

flags.DEFINE_string('model_name', None, 'Model name - one of configs.')
flags.DEFINE_string('exp_name', None, 'Experiment name to load from.')

FLAGS = flags.FLAGS


MODEL_DIR = '/media/14tb/ml/models/zetaqubit/dl/examples/wikitext'


def load_model(dir):
  tokenizer = hf_transformers.AutoTokenizer.from_pretrained('gpt2')
  tokenizer.pad_token = tokenizer.eos_token
  model = transformer.AutoregressiveModel(tokenizer=tokenizer)
  checkpoint.load_ckpt(dir, model)
  # model.load_state_dict(torch.load(path))
  model.eval()
  model.cuda()
  return model, tokenizer


def generate(_):
  exp_dir = os.path.join(MODEL_DIR, FLAGS.model_name, FLAGS.exp_name)
  gin.parse_config_file(f'{exp_dir}/config.gin')

  model, tokenizer = load_model(exp_dir)

  while True:
    try:
      prompt = input(f'Prompt: ')
    except (EOFError, KeyboardInterrupt):
      print()
      break
    out_text = model.generate(prompt,
                              seq_len=gin.query_parameter('%max_seq_len'))
    print(out_text)


if __name__ == '__main__':
  app.run(generate)
  flags.mark_flags_as_required(['model_name', 'exp_name'])
