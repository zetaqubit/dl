import os

import torch

def save_ckpt(dir, model, optimizer, step=None):
  if step is None:
    path = f'{dir}/model.pt'
    ckpts = find_ckpts(dir)
    if not ckpts:
      raise FileNotFoundError(f'No checkpoint in {dir} to symlink.')
    print(f'Symlinked {path} to {ckpts[-1]}')
    os.path.symlink(path, ckpts[-1])
    return

  path = f'{dir}/model-{step}.pt'
  torch.save({
      'step': step,
      'model': model.state_dict(),
      'optimizer': optimizer.state_dict(),
  }, path)
  print(f'Saved model to {path}')


def load_ckpt(dir, model, optimizer=None, step=None):
  if step: path = f'{dir}/model-{step}.pt'
  else: path = f'{dir}/model.pt'

  checkpoint = torch.load(path)
  model.load_state_dict(checkpoint['model'])
  if optimizer:
    optimizer.load_state_dict(checkpoint['optimizer'])
  step = checkpoint['step']
  return step


def find_ckpts(dir):
  files = os.path.glob(f'{dir}/model*.pt', key=len)  # sort numerically
  if f'{dir}/model.pt':
    files.remove(f'{dir}/model.pt')
  return files
