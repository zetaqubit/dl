"""Library for training transformer on wikitext.

No FLAGS to allow it to be imported (e.g. by optuna).
All configuration has happened a priori via gin.
"""

import math
import os
import shutil
import signal

from einops import rearrange
import gin
import numpy as np
import pandas as pd
import torch
from torch.utils import tensorboard as tb
import torchinfo
import tqdm

from dl.data import dataset
from dl.data import tokenizers
from dl.examples.wikitext import checkpoint
from dl.models import models
from dl.rnn import rnn
from dl.utils.config_utils import gin_get


def cycle(loader):
  while True:
    for data in loader:
      yield data


@torch.no_grad()
def estimate_loss(model, data_loader, steps, **kwargs):
  model.eval()
  losses = torch.zeros(steps)
  for i, ids in zip(range(steps), data_loader):
    losses[i] = model(ids.to('cuda'), **kwargs).item()
  model.train()
  return losses.mean()

def filter_example(example):
  text = example['text']
  return len(text) > 64 and not text.startswith(' =')


_TEXT_SUMMARY = '''
#### generated
{}
#### ground truth
{}
#### mle
{}
'''

@torch.no_grad()
def text_completion_sxs(model, text, num=2):
  # Example text and generation.
  n_prompt = 16
  words = text.split(' ')
  if len(words) > 2 * n_prompt:
    prompt, gt = ' '.join(words[:n_prompt]), ' '.join(words[n_prompt:])
  else:
    prompt = text[:32]
  generated = prompt + model.generate_text(prompt=prompt, seq_len=128)
  tok = model.tokenizer
  in_ids = torch.tensor(tok.encode(text)).to('cuda')
  in_ids = rearrange(in_ids, 's -> 1 s')
  mle_ids = torch.argmax(model.net(in_ids), dim=-1)
  mle = tok.decode_batch(mle_ids)[0]
  log_ex = _TEXT_SUMMARY.format(generated, text, mle)
  return log_ex


def decode_ids(tokenizer, ids):
  mask = ~(ids == tokenizer.padding_id)
  mask[0] = 1  # keep at least the first token, even if it's padding.
  ids = ids[mask]  # remove trailing padding
  text = tokenizer.decode(ids)
  return text


def gradient_stats(model):
  l2_norm = 0
  zero_frac = 0
  total_params = 0
  for param in model.parameters():
    param_norm = param.grad.data.norm(2)
    l2_norm += param_norm.item() ** 2
    zero_frac += (param.grad == 0).sum().item()
    total_params += param.grad.numel()
  l2_norm = l2_norm ** (1. / 2)
  zero_frac /= total_params
  return {
    'l2_norm': l2_norm,
    'zero_frac': zero_frac,
  }


MODEL_DIR = '/media/14tb/ml/models/zetaqubit/dl/examples/wikitext'


def train():
  torch.manual_seed(42)
  print(gin.config_str())

  model_name = gin_get('%model_name')
  exp_name = gin_get('%exp_name')
  exp_dir = os.path.join(MODEL_DIR, model_name, exp_name)
  resume = gin_get('%resume')

  if not resume:
    shutil.rmtree(exp_dir, ignore_errors=True)
  os.makedirs(exp_dir, exist_ok=True)

  max_seq_len = gin_get('%max_seq_len')

  tok_type = gin_get('tokenizers.create.tok_type')
  tokenizer = tokenizers.create()

  ds_name = gin_get('%dataset')
  print(f'Loading dataset {ds_name}/{tok_type}.train.bin')
  ds_train = dataset.MemoryMappedDataset(
      name=ds_name, split=f'{tok_type}.train', block_size=max_seq_len+1,
      padding_token=tokenizer.padding_id)
  ds_valid = dataset.MemoryMappedDataset(
      name=ds_name, split=f'{tok_type}.val', block_size=max_seq_len+1,
      padding_token=tokenizer.padding_id)

  batch_size = gin_get('%batch_size')
  dl_train = torch.utils.data.DataLoader(
      dataset=ds_train, batch_size=batch_size)
  dl_valid = torch.utils.data.DataLoader(
      dataset=ds_valid, batch_size=batch_size)
  iter_train = cycle(dl_train)

  ids_valid = next(iter(dl_valid))
  text_valid = decode_ids(tokenizer, ids_valid[0, :-1])

  train_steps = gin_get('%train_steps')
  log_steps = gin_get('%log_steps')
  eval_interval = gin_get('%eval_interval')
  eval_steps = gin_get('%eval_steps')
  ckpt_steps = gin_get('%ckpt_steps')

  model = models.create()
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
      optim, total_steps=train_steps, warmup_steps=int(0.1 * train_steps))

  # min_lr = 1e-7
  # terminating_lr = 1.01e-6  # once LR reaches this point, stop the training.
  # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
  #   optim, mode='min', factor=0.3, patience=300, cooldown=100,
  #   min_lr=min_lr)

  writer = tb.SummaryWriter(exp_dir)
  writer.add_text('model/gin_config', gin.markdown(gin.config_str()), 0)
  model_summary = torchinfo.summary(model, input_data=ids_valid)
  model_summary = '\n'.join([f'    {s}' for s in str(model_summary).split('\n')])
  writer.add_text('model/summary', model_summary)

  # Write vocab.
  # with open(os.path.join(exp_dir, 'vocab.txt'), 'w') as fd:
  #   vocab = {v: k for k, v in tokenizer.get_vocab().items()}
  #   vocab = dict(sorted(vocab.items(), key=lambda item: item[0]))
  #   df = pd.DataFrame.from_dict(vocab, orient='index')
  #   df.to_csv(fd, index=False)

  # Write config.gin
  with open(os.path.join(exp_dir, 'config.gin'), 'w') as fd:
    fd.write(gin.config_str())

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
    del state

  accum_grad_steps = gin_get('%accum_grad_steps', 1)
  record_gradients = gin_get('%record_gradients', True)
  max_grad_norm = 0
  batches_skipped = 0

  pbar = tqdm.tqdm(range(start_step, end_step + 1), desc='training',
                   mininterval=0.5)
  for i in pbar:
    avg_loss = 0
    for _ in range(accum_grad_steps):
      ids = next(iter_train)
      loss = model(ids.to('cuda'))
      # loss += model(ids.to('cuda'), teacher_forcing='first_half')
      # loss /= 2
      loss /= accum_grad_steps
      loss.backward()
      avg_loss += loss.item()
      tokens_seen += (ids.detach() != model.ignore_index).sum().item()
      tokens_total += ids.shape[0] * ids.shape[1]

    pbar.set_description(f'train loss: {avg_loss:.2f}')

    stop_training = (i == end_step)
    if stop_training or record_gradients or (i % log_steps == 0):
      writer.add_scalar('step', i, i)
      writer.add_scalar('loss/train', avg_loss, i)
      lr = optim.param_groups[0]['lr']
      writer.add_scalar('learning_rate', lr, i)
      writer.add_scalar('step/tokens_seen', tokens_seen, i)
      writer.add_scalar('step/tokens_used_rate', tokens_seen / tokens_total, i)
      writer.add_scalar('over_tokens/loss_train', avg_loss, tokens_seen)
      writer.flush()
      # if lr <= terminating_lr:
      #   stop_training = True
      #   print(f'Training terminating early at step {i}, '
      #         f'lr = {lr}, loss = {avg_loss}')

    if record_gradients:
      grad_stats = gradient_stats(model)
      for name, stat in grad_stats.items():
        writer.add_scalar(f'grad/{name}', stat, i)
      writer.add_scalar('grad/batches_skipped', batches_skipped, i)

      grad_norm = grad_stats['l2_norm']
      if max_grad_norm and grad_norm > 5 * max_grad_norm:
        batches_skipped += 1
        lr_scheduler.step()
        optim.zero_grad()
        continue
      max_grad_norm = max(max_grad_norm, grad_norm)

    optim.step()
    optim.zero_grad()
    lr_scheduler.step()



    if stop_training or (i % eval_interval == 0):
      loss_train = estimate_loss(model, dl_train, eval_steps)
      loss_valid = estimate_loss(model, dl_valid, eval_steps)
      writer.add_scalar('eval/loss_train', loss_train, i)
      writer.add_scalar('eval/loss_valid', loss_valid, i)
      writer.add_scalar('eval/perplexity_valid', loss_valid.exp(), i)
      writer.add_scalar('eval/bits_per_token_valid', loss_valid / np.log(2), i)

      writer.add_scalar('over_tokens/loss_eval_train', loss_train, tokens_seen)
      writer.add_scalar('over_tokens/loss_eval_valid', loss_valid, tokens_seen)

      if isinstance(model, rnn.GenerativeRnnModel):
        teacher_forcing = ['all', 'first_half']
        for mode in teacher_forcing:
          loss_mode = estimate_loss(model, dl_train, eval_steps,
                                    teacher_forcing=mode)
          writer.add_scalar(f'eval/loss_train_force_{mode}', loss_mode, i)

      text = decode_ids(tokenizer, ids[0, :-1])
      log_ex = text_completion_sxs(model, text)
      writer.add_text('example/train', log_ex, i)

      log_ex = text_completion_sxs(model, text_valid)
      writer.add_text('example/eval', log_ex, i)

    if stop_training or (i % ckpt_steps == 0):
      checkpoint.save_ckpt(exp_dir, model, optim, step=i,
                           tokens_seen=tokens_seen,
                           tokens_total=tokens_total)
      checkpoint.save_ckpt(exp_dir, model, optim)  # symlink to latest

    if stop_training: break

  post_training(None, None)

  # Return final metrics
  return {
    'loss/train': loss.item(),
    'eval/loss_train': loss_train,
    'eval/loss_valid': loss_valid,
    'step': i,
  }
