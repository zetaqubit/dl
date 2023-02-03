"""Trains transformer on wikitext."""

import datasets as hf_datasets
import gin
import transformers as hf_transformers
import torch
import tqdm

from dl.transformer import transformer

def cycle(loader):
    while True:
        for data in loader:
            yield data

gin.parse_config_file('dl/examples/wikitext/config.gin')

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
pbar = tqdm.tqdm(range(train_steps), mininterval=10., desc='training')
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
