"""Generates text based on a prompt."""

import gin
import torch
import transformers as hf_transformers

from dl.transformer import transformer

config_path = 'dl/examples/wikitext/configs/gpt-8l-768d-128msl.gin'
gin.parse_config_file(config_path)

model_path = (
  '/media/14tb/ml/models/zetaqubit/dl/examples/wikitext/'
  f'{gin.query_parameter("%exp_name")}/model.pt'
)

def load_model(path):
  model = transformer.AutoregressiveModel()
  model.load_state_dict(torch.load(path))
  model.eval()
  model.cuda()
  tokenizer = hf_transformers.AutoTokenizer.from_pretrained('gpt2')
  tokenizer.pad_token = tokenizer.eos_token
  return model, tokenizer

model, tokenizer = load_model(model_path)

while True:
  try:
    prompt = input(f'Prompt: ')
  except (EOFError, KeyboardInterrupt):
    print()
    break
  ids = tokenizer(prompt, return_tensors='pt')['input_ids'].to('cuda')
  out_ids = model.generate(ids, seq_len=128)
  out_text = tokenizer.decode(out_ids[0, ...])
  print(out_text)

