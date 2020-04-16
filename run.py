# Copyright (c) 2020 smarsu. All Rights Reserved.

from model import NCFModel
import numpy as np
np.random.seed(196)


def train():
  """Train a ncf-model with ml-25m dataset."""
  model = NCFModel(device='cuda')
  # Need large batch size.
  model.train(epoch=10, batch_size=1024, lr=0.01)


if __name__ == '__main__':
  train()
