import glob
import os
import pathlib
import tempfile

import torch

def save_ckpt(dir, model, optimizer, step=None, **kwargs):
  if step is None:
    path = f'{dir}/model.pt'
    ckpts = find_ckpts(dir)
    if not ckpts:
      raise FileNotFoundError(f'No checkpoint in {dir} to symlink.')
    symlink_force(ckpts[-1], path)
    print(f'Symlinked {path} -> {ckpts[-1]}')
    return

  path = f'{dir}/model-{step}.pt'
  state = {
      'step': step,
      'model': model.state_dict(),
      'optimizer': optimizer.state_dict(),
  }
  state.update(kwargs)
  torch.save(state, path)
  print(f'Saved model to {path}')


def load_ckpt(dir, model, optimizer=None, step=None):
  """Loads the checkpoint at a specific step."""
  if step: path = f'{dir}/model-{step}.pt'
  else: path = f'{dir}/model.pt'
  print(f'Loading from {path} ({os.path.realpath(path)})')

  state = torch.load(path)
  model.load_state_dict(state['model'])
  if optimizer:
    optimizer.load_state_dict(state['optimizer'])
  return state


def find_ckpts(dir):
  """Returns paths to the checkpoints in dir, excluding the symlink model.pt"""
  files = glob.glob(f'{dir}/model*.pt')
  files.sort(key=os.path.getmtime)  # sort by modification time.
  try: files.remove(f'{dir}/model.pt')
  except ValueError: pass
  return files

def symlink_force(src, link_name):
  with tempfile.TemporaryDirectory(dir=os.path.dirname(link_name)) as d:
    tmpname = os.path.join(d, "foo")
    os.symlink(src, tmpname)
    os.replace(tmpname, link_name)