"""Generates text based on a prompt."""
import argparse

import gin
import torch
import transformers as hf_transformers

from dl.transformer import transformer

parser = argparse.ArgumentParser()
parser.add_argument('--gin_config',
                    default='gpt-8l-768d-128msl.gin',
                    help='Path to the config.gin, relative to configs/')
args = parser.parse_args()

gin.parse_config_file(f'dl/examples/wikitext/configs/{args.gin_config}')

model_path = (
  '/media/14tb/ml/models/zetaqubit/dl/examples/wikitext/'
  f'{gin.query_parameter("%exp_name")}/model.pt'
)

def load_model(path):
  tokenizer = hf_transformers.AutoTokenizer.from_pretrained('gpt2')
  tokenizer.pad_token = tokenizer.eos_token
  model = transformer.AutoregressiveModel(tokenizer=tokenizer)
  model.load_state_dict(torch.load(path))
  model.eval()
  model.cuda()
  return model, tokenizer

model, tokenizer = load_model(model_path)

while True:
  try:
    prompt = input(f'Prompt: ')
  except (EOFError, KeyboardInterrupt):
    print()
    break
  out_text = model.generate(prompt, seq_len=gin.query_parameter('%max_seq_len'))
  print(out_text)

