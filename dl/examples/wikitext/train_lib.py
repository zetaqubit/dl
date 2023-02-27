"""Library for training transformer on wikitext.

No FLAGS to allow it to be imported (e.g. by optuna).
All configuration has happened a priori via gin.
"""

import math
import os
import shutil
import signal

import datasets as hf_datasets
import gin
import numpy as np
import pandas as pd
import transformers as hf_transformers
import torch
from torch.utils import tensorboard as tb
import torchinfo
import tqdm

from dl.examples.wikitext import checkpoint
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

def gin_get(param, default=None):
  try:
    return gin.query_parameter(param)
  except ValueError as e:
    if default: return default
    raise

@torch.no_grad()
def text_completion_sxs(model, texts, num=2):
  # Example text and generation.
  form = '''
    | prompt       | {} |
    | ground truth | {} |
    | generated    | {} |
  '''
  text = texts[0]
  text = text.rstrip(' \n')
  words = text.split(' ')
  prompt, gt = ' '.join(words[:16]), ' '.join(words[16:])
  generated = model.generate(prompt, 128)
  generated = generated.strip('\n')
  log_ex = form.format(prompt, gt, generated)
  return log_ex



MODEL_DIR = '/media/14tb/ml/models/zetaqubit/dl/examples/wikitext'


def train():
  model_name = gin_get('%model_name')
  exp_name = gin_get('%exp_name')
  exp_dir = os.path.join(MODEL_DIR, model_name, exp_name)
  resume = gin_get('%resume')

  if not resume:
    shutil.rmtree(exp_dir, ignore_errors=True)
  os.makedirs(exp_dir, exist_ok=True)

  max_seq_len = gin_get('%max_seq_len')

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

  batch_size = gin_get('%batch_size')
  dl_train = torch.utils.data.DataLoader(
    dataset=ds_train, batch_size=batch_size) #, shuffle=True)
  dl_valid = torch.utils.data.DataLoader(
    dataset=ds_valid, batch_size=batch_size) #, shuffle=False)
  iter_train = cycle(dl_train)

  train_steps = gin_get('%train_steps')
  log_steps = gin_get('%log_steps')
  eval_interval = gin_get('%eval_interval')
  eval_steps = gin_get('%eval_steps')
  ckpt_steps = gin_get('%ckpt_steps')

  model = transformer.AutoregressiveModel(tokenizer=tokenizer)
  model.cuda()
  optim = torch.optim.Adam(model.parameters(),
                          lr=gin_get('%learning_rate'))
  # optim = torch.optim.RMSprop(model.parameters(),
  #                             lr=gin_get('%learning_rate'),
  #                             alpha=0.9)
  # optim = torch.optim.SGD(model.parameters(), lr=gin_get('%learning_rate'),
  #                         momentum=0.9)


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

  # Cleanup after training
  def post_training(signum, frame):
    # Symlink final model.
    checkpoint.save_ckpt(exp_dir, model, optim)
    writer.close()
    if signum: exit(1)
  signal.signal(signal.SIGINT, post_training)

  tokens_seen = 0
  tokens_total = 0
  start_step = 0
  end_step = train_steps
  if resume:
    state = checkpoint.load_ckpt(exp_dir, model,
                                 optim if resume == 'model_opt' else None)
    start_step = state['step']
    tokens_seen = state.get('tokens_seen', 0)
    tokens_total = state.get('tokens_total', 0)
    end_step += start_step

  accum_grad_steps = gin_get('%accum_grad_steps', 1)

  pbar = tqdm.tqdm(range(start_step, end_step + 1), desc='training',
                   mininterval=0.5)
  for i in pbar:
    for _ in range(accum_grad_steps):
      ex = next(iter_train)
      text, ids = ex['text'], ex['ids']  # [b, s]
      loss = model(ids.to('cuda'))
      (loss / accum_grad_steps).backward()
      tokens_seen += (ids.detach() != model.ignore_index).sum().item()
      tokens_total += ids.shape[0] * ids.shape[1]

    optim.step()
    optim.zero_grad()
    lr_scheduler.step()

    pbar.set_description(f'train loss: {loss.item():.2f}')


    stop_training = (i == end_step)
    if stop_training or (i % log_steps == 0):
      writer.add_scalar('step', i, i)
      writer.add_scalar('loss/train', loss.item(), i)
      lr = optim.param_groups[0]['lr']
      writer.add_scalar('learning_rate', lr, i)
      writer.add_scalar('step/tokens_seen', tokens_seen, i)
      writer.add_scalar('step/tokens_used_rate', tokens_seen / tokens_total, i)
      writer.add_scalar('over_tokens/loss_train', loss.item(), tokens_seen)
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
      writer.add_scalar('eval/perplexity_valid', loss_valid.exp(), i)
      writer.add_scalar('eval/bits_per_token_valid', loss_valid / np.log(2), i)

      writer.add_scalar('over_tokens/loss_eval_train', loss_train, tokens_seen)
      writer.add_scalar('over_tokens/loss_eval_valid', loss_valid, tokens_seen)

      log_ex = text_completion_sxs(model, text)
      writer.add_text('example/train', log_ex, i)

      log_ex = text_completion_sxs(model, next(iter(dl_valid))['text'])
      writer.add_text('example/eval', log_ex, i)


    if stop_training or (i % ckpt_steps == 0):
      checkpoint.save_ckpt(exp_dir, model, optim, step=i,
                           tokens_seen=tokens_seen,
                           tokens_total=tokens_total)

    if stop_training: break

  # Write vocab.
  with open(os.path.join(exp_dir, 'vocab.txt'), 'w') as fd:
    vocab = {v: k for k, v in tokenizer.get_vocab().items()}
    vocab = dict(sorted(vocab.items(), key=lambda item: item[0]))
    df = pd.DataFrame.from_dict(vocab, orient='index')
    df.to_csv(fd, index=False)

  # Write config.gin
  with open(os.path.join(exp_dir, 'config.gin'), 'w') as fd:
    fd.write(gin.operative_config_str())

  post_training(None, None)

  # Return final metrics
  return {
    'loss/train': loss.item(),
    'eval/loss_train': loss_train,
    'eval/loss_valid': loss_valid,
    'step': i,
  }
