"""Library for training transformer on wikitext.

No FLAGS to allow it to be imported (e.g. by optuna).
All configuration has happened a priori via gin.
"""

import math
import os

import datasets as hf_datasets
import gin
import pandas as pd
import transformers as hf_transformers
import torch
from torch.utils import tensorboard as tb
import torchinfo
import tqdm

from dl.transformer import transformer


def cycle(loader):
    while True:
        for data in loader:
            yield data

@torch.no_grad()
def estimate_loss(model, data_loader, steps):
  model.eval()
  losses = torch.zeros(steps)
  for i, batch in zip(range(steps), data_loader):
    text, ids = batch['text'], batch['ids']
    losses[i] = model(ids.to('cuda')).item()
  model.train()
  return losses.mean()

def filter_example(example):
  text = example['text']
  return len(text) > 64 and not text.startswith(' =')



MODEL_DIR = '/media/14tb/ml/models/zetaqubit/dl/examples/wikitext'


def train():
  model_name = gin.query_parameter('%model_name')
  exp_name = gin.query_parameter('%exp_name')
  exp_dir = os.path.join(MODEL_DIR, model_name, exp_name)
  os.makedirs(exp_dir, exist_ok=True)

  max_seq_len = gin.query_parameter('%max_seq_len')

  tokenizer = hf_transformers.AutoTokenizer.from_pretrained('gpt2')
  tokenizer.pad_token = tokenizer.eos_token

  def tokenize(example):
    text = example['text']
    tokenized = tokenizer(
      text, padding='max_length', truncation=True,
      max_length=max_seq_len,
      return_tensors='pt')
    example['ids'] = tokenized['input_ids'][0]
    return example


  ds = hf_datasets.load_dataset(path='wikitext', name='wikitext-103-v1',
                                streaming=True)
  ds = ds.filter(filter_example)
  ds = ds.map(tokenize)
  ds = ds.with_format('torch')
  ds_train, ds_valid = ds['train'], ds['validation']

  batch_size = gin.query_parameter('%batch_size')
  dl_train = torch.utils.data.DataLoader(
    dataset=ds_train, batch_size=batch_size) #, shuffle=True)
  dl_valid = torch.utils.data.DataLoader(
    dataset=ds_valid, batch_size=batch_size) #, shuffle=False)
  iter_train, iter_valid = cycle(dl_train), cycle(dl_valid)

  train_steps = gin.query_parameter('%train_steps')
  log_steps = gin.query_parameter('%log_steps')
  eval_interval = gin.query_parameter('%eval_interval')
  eval_steps = gin.query_parameter('%eval_steps')

  model = transformer.AutoregressiveModel(tokenizer=tokenizer)
  model.cuda()
  optim = torch.optim.Adam(model.parameters(),
                          lr=gin.query_parameter('%learning_rate'))

  def get_lr_scheduler(optimizer, total_steps, warmup_steps):
      def lr_lambda(step):
          if step < warmup_steps:
              return float(step) / float(max(1, warmup_steps))
          return 0.5 * (1.0 + math.cos(math.pi * float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))))

      return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

  lr_scheduler = get_lr_scheduler(
      optim, total_steps=train_steps, warmup_steps=1500)

  # min_lr = 1e-7
  # terminating_lr = 1.01e-6  # once LR reaches this point, stop the training.
  # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
  #   optim, mode='min', factor=0.3, patience=300, cooldown=100,
  #   min_lr=min_lr)

  writer = tb.SummaryWriter(exp_dir)
  writer.add_text('model/gin_config', gin.markdown(gin.operative_config_str()), 0)
  model_summary = torchinfo.summary(
    model, input_data=next(iter(dl_train))['ids'].to('cuda'))
  model_summary = '\n'.join([f'    {s}' for s in str(model_summary).split('\n')])
  writer.add_text('model/summary', model_summary)

  pbar = tqdm.tqdm(range(train_steps + 1), desc='training')
  for i in pbar:
    ex = next(iter_train)
    text, ids = ex['text'], ex['ids']  # [b, s]
    loss = model(ids.to('cuda'))
    loss.backward()
    optim.step()
    optim.zero_grad()
    lr_scheduler.step()

    stop_training = False
    if i % log_steps == 0:
      writer.add_scalar('step', i, i)
      writer.add_scalar('loss/train', loss.item(), i)
      lr = optim.param_groups[0]['lr']
      writer.add_scalar('learning_rate', lr, i)
      writer.flush()
      # if lr <= terminating_lr:
      #   stop_training = True
      #   print(f'Training terminating early at step {i}, '
      #         f'lr = {lr}, loss = {loss.item()}')

    if stop_training or (i % eval_interval == 0):
      loss_train = estimate_loss(model, dl_train, eval_steps)
      loss_valid = estimate_loss(model, dl_valid, eval_steps)
      writer.add_scalar('eval/loss_train', loss_train, i)
      writer.add_scalar('eval/loss_valid', loss_valid, i)

      # Example text and generation.
      form = '''
      | prompt       | {} |
      | ground truth | {} |
      | generated    | {} |
      '''
      text = text[0].rstrip(' \n')
      words = text.split(' ')
      prompt, gt = ' '.join(words[:32]), ' '.join(words[32:])
      generated = model.generate(prompt, 128)
      log_ex = form.format(prompt, gt, generated)
      writer.add_text('example/generated', log_ex, i)


    if i % 10 == 0:
      pbar.set_description(f'train loss: {loss.item():.2f}')

    if stop_training: break

  # Write vocab.
  with open(os.path.join(exp_dir, 'vocab.txt'), 'w') as fd:
    vocab = {v: k for k, v in tokenizer.get_vocab().items()}
    vocab = dict(sorted(vocab.items(), key=lambda item: item[0]))
    df = pd.DataFrame.from_dict(vocab, orient='index')
    df.to_csv(fd, index=False)

  # Write model.
  model_path = os.path.join(exp_dir, 'model.pt')
  torch.save(model.state_dict(), model_path)
  print(f'Saved model to {model_path}')

  # Write config.gin
  with open(os.path.join(exp_dir, 'config.gin'), 'w') as fd:
    fd.write(gin.operative_config_str())

  writer.close()

  # Return final metrics
  return {
    'loss/train': loss.item(),
    'eval/loss_train': loss_train,
    'eval/loss_valid': loss_valid,
    'step': i,
  }