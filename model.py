# Copyright (c) 2020 smarsu. All Rights Reserved.

import os.path as osp
from tqdm import tqdm
import torch
import numpy as np

from ncf import NCF
from datasets import ML25M


class NCFModel(object):
  def __init__(self, restore_path=None, device='cpu'):
    # self.mf_dim = 64
    # self.mlp_layer_sizes = [256, 256, 128, 64]

    self.mf_dim = 8
    self.mlp_layer_sizes = [32, 32, 16, 8]

    self.device = torch.device(device)

    self.ml_25m = ML25M()
    self.ncf = NCF(self.ml_25m.num_users, 
                   self.ml_25m.num_items, 
                   self.mf_dim, 
                   self.mlp_layer_sizes, 
                   self.device)
    self.ncf.to(self.device)
    if restore_path is not None:
      self.ncf.load_state_dict(torch.load(restore_path))


  def preprocess(self, rating):
    """Rating is between [0.5, 5], normalize it to [0, 1]

    Args:
      rating: float

    Returns:
      rating: float
    """
    # No converge
    # return (rating - 0.5) / 4.5
    return (rating > 2.5).astype(np.float32)


  def train(self, epoch, batch_size, lr=0.1, momentum=0.9, weight_decay=4e-5, save_root=None):
    for v in self.ncf.parameters():
      print(v.name, v.size())
      print(v)
    # sgd = torch.optim.SGD(self.ncf.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    # Need Adam optimizer
    sgd = torch.optim.Adam(self.ncf.parameters(), weight_decay=weight_decay)

    for step in range(epoch):
      pbar = tqdm(list(self.ml_25m.GetTrainEpoch(batch_size)))

      sum_loss = 0
      cnt = 0
      for train_label_batch in pbar:
        assert train_label_batch.shape == (batch_size, 3), train_label_batch.shape

        user_id, movie_id, rating = [train_label_batch[:, i] for i in range(3)]
        user_id = user_id.astype(np.int64)
        movie_id = movie_id.astype(np.int64)
        rating = rating.reshape(-1, 1)
        rating = self.preprocess(rating)

        user_id = torch.from_numpy(user_id).to(self.device)
        movie_id = torch.from_numpy(movie_id).to(self.device)
        rating = torch.from_numpy(rating).to(self.device)

        self.ncf = self.ncf.train()
        logits = self.ncf(user_id, movie_id)
        loss = self.ncf.ncf_loss(logits, rating)

        sgd.zero_grad()
        loss.backward()
        sgd.step()

        sum_loss += loss.cpu().detach().numpy()
        cnt += 1

        avg_loss = sum_loss / cnt
        pbar.set_description('step: {}, loss: {}'.format(step, avg_loss))

      self.eval()

      if save_root is not None:
        torch.save(
          self.ncf.state_dict(), 
          osp.join(save_root, '{}-{}-{}'.format(step, lr, avg_loss)))


  def compute_eval_metrics(self, logits, lables):
    """Compute the Hit Rate and NDCG.
    
    Args:
      logits: ndarray.
      lables: ndarray.
    """
    n_sum = logits.size
    n_tp = np.sum(np.logical_and(logits > 0, lables > 2.5))
    n_tn = np.sum(np.logical_and(logits <= 0, lables <= 2.5))
    return (n_tp + n_tn), n_sum


  def eval(self):
    n_tp_tn_sum = 0
    n_sum = 0
    pbar = tqdm(self.ml_25m.test_labels)
    for test_lable in pbar:
      user_id, movie_id, rating = test_lable

      user_id = torch.tensor([int(user_id)]).to(self.device)
      movie_id = torch.tensor([int(movie_id)]).to(self.device)
      rating = torch.tensor([rating]).to(self.device)

      self.ncf = self.ncf.eval()
      logits = self.ncf(user_id, movie_id)

      n_tp_tn, n = self.compute_eval_metrics(
        logits.cpu().detach().numpy(), rating.cpu().detach().numpy())

      n_tp_tn_sum += n_tp_tn
      n_sum += n

      pbar.set_description(
        'right: {}, sum: {}, right-rate: {}'.format(
          n_tp_tn_sum, n_sum, n_tp_tn_sum / n_sum))
