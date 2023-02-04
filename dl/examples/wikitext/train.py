"""Trains transformer on wikitext."""

import argparse
import os

import datasets as hf_datasets
import gin
import pandas as pd
import transformers as hf_transformers
import torch
import tqdm

from dl.transformer import transformer

def cycle(loader):
    while True:
        for data in loader:
            yield data


parser = argparse.ArgumentParser()
parser.add_argument('--gin_config',
                    default='gpt-8l-768d-128msl.gin',
                    help='Path to the config.gin, relative to configs/')
args = parser.parse_args()

gin.parse_config_file(f'dl/examples/wikitext/configs/{args.gin_config}')

ds = hf_datasets.load_dataset(path='wikitext', name='wikitext-103-v1')
ds = ds.with_format('torch', device='cuda')
ds_train, ds_valid = ds['train'], ds['validation']

batch_size = gin.query_parameter('%batch_size')
dl_train = torch.utils.data.DataLoader(
  dataset=ds_train, batch_size=batch_size, shuffle=True)
dl_valid = torch.utils.data.DataLoader(
  dataset=ds_valid, batch_size=batch_size, shuffle=False)
dl_train, dl_valid = cycle(dl_train), cycle(dl_valid)

# TODO: dataset uses <unk>. Check that tokenizer is compatible.
tokenizer = hf_transformers.AutoTokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token
model = transformer.AutoregressiveModel()
model.cuda()

optim = torch.optim.Adam(model.parameters(),
                         lr=gin.query_parameter('%learning_rate'))

train_steps = gin.query_parameter('%train_steps')
pbar = tqdm.tqdm(range(train_steps), desc='training')
for i in pbar:
  text = next(dl_train)['text']
  tokenized = tokenizer(
    text, padding='max_length', truncation=True,
    max_length=gin.query_parameter('%max_seq_len'),
    return_tensors='pt')
  ids = tokenized['input_ids'].to('cuda')
  loss = model(ids)
  loss.backward()
  optim.step()
  optim.zero_grad()

  if i % 10 == 0:
    pbar.set_description(f'train loss: {loss.item():.2f}')


model_dir = '/media/14tb/ml/models/zetaqubit/dl/examples/wikitext'
exp_name = gin.query_parameter('%exp_name')
exp_dir = os.path.join(model_dir, exp_name)
os.makedirs(exp_dir, exist_ok=True)

# Write vocab.
with open(os.path.join(exp_dir, 'vocab.txt'), 'w') as fd:
  vocab = {v: k for k, v in tokenizer.get_vocab().items()}
  vocab = dict(sorted(vocab.items(), key=lambda item: item[0]))
  df = pd.DataFrame.from_dict(vocab, orient='index')
  df.to_csv(fd, index=False)

# Write model
model_path = os.path.join(exp_dir, 'model.pt')
torch.save(model.state_dict(), model_path)
print(f'Saved model to {model_path}')