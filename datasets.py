# Copyright (c) 2020 smarsu. All Rights Reserved.

import os.path as osp
from tqdm import tqdm
import numpy as np


class ML25M(object):
  def __init__(self, root='/datasets/ml-25m'):
    """
    Args:
      root: str. The root of ml-25m/ml-latest-small dataset.
    """
    self._root = root

    (self.train_labels_pos, self.test_labels_pos, 
     self.train_labels_neg, self.test_labels_neg, 
     self.user2movie, self.movie2user) = self._load(
      self._root)

    lpos = len(self.train_labels_pos)
    lneg = len(self.train_labels_neg)
    if lpos > lneg:
      factor = int(lpos / lneg)
      self.train_labels_neg = np.concatenate([self.train_labels_neg] * factor, 0)
    if lneg > lpos:
      factor = int(lneg / lpos)
      self.train_labels_pos = np.concatenate([self.train_labels_pos] * factor, 0)

    print(self.train_labels_pos.shape)
    print(self.test_labels_pos.shape)
    print(self.train_labels_neg.shape)
    print(self.test_labels_neg.shape)
    print(self.test_labels_pos)
    print(self.test_labels_neg)
    self.test_labels = np.concatenate([self.test_labels_pos, self.test_labels_neg], 0)
    self.num_users = len(self.user2movie)
    self.num_items = len(self.movie2user)


  def GetTrainEpoch(self, batch_size):
    """
    Args:
      batch_size: int
    """
    # self.train_labels_pos = self.train_labels_pos[:batch_size//2]
    # self.train_labels_neg = self.train_labels_neg[:batch_size//2]

    # length = min(len(self.train_labels_pos), len(self.train_labels_neg))
    # np.random.shuffle(self.train_labels_pos)
    # np.random.shuffle(self.train_labels_neg)
    # self.train_labels = np.concatenate([self.train_labels_pos[:length], self.train_labels_neg[:length]], 0)

    self.train_labels = np.concatenate([self.train_labels_pos, self.train_labels_neg], 0)

    num_labels = len(self.train_labels)
    ids = np.arange(num_labels)
    np.random.shuffle(ids)

    ids = ids[:num_labels // batch_size * batch_size]
    ids = ids.reshape(-1, batch_size)

    train_lables_shuffle = self.train_labels[ids]
    for train_label_batch in train_lables_shuffle:
      yield train_label_batch


  def _load(self, root):
    """
    Args:
      root: str. The root of ml-25m dataset.
    """
    path = osp.join(root, 'ratings.csv')

    user2movie = {}
    movie2user = {}
    movie2id = {}
    movie_count = 0
    train_labels_pos = []
    train_labels_neg = []
    with open(path, 'r') as fb:
      lines = fb.readlines()[1:]  # Drop the first line which is tag
      pbar = tqdm(lines)
      for line in pbar:
        user_id, movie_id, rating, timestamp = line.split(',')
        user_id, movie_id, rating, timestamp = [
          float(v) for v in [user_id, movie_id, rating, timestamp]]
        # Begin from 0
        user_id -= 1
        movie_id -= 1

        if movie_id not in movie2id:
          movie2id[movie_id] = movie_count
          movie_count += 1

        movie_id = movie2id[movie_id]

        # train_labels.append([user_id, movie_id, rating])
        if rating > 2.5:
          train_labels_pos.append([user_id, movie_id, rating])
        else:
          train_labels_neg.append([user_id, movie_id, rating])

        if user_id not in user2movie:
          user2movie[user_id] = [movie_id]
        else:
          user2movie[user_id].append(movie_id)
        if movie_id not in movie2user:
          movie2user[movie_id] = [user_id]
        else:
          movie2user[movie_id].append(user_id)

        # pbar.set_description('{}/{}'.format(len(train_labels), len(lines)))

    train_labels_pos = np.array(train_labels_pos, dtype=np.float32)
    train_labels_neg = np.array(train_labels_neg, dtype=np.float32)
    np.random.shuffle(train_labels_pos)
    np.random.shuffle(train_labels_neg)

    length = min(len(train_labels_pos), len(train_labels_neg))
    num_test_labels = round(length * 0.01)

    test_labels_pos = train_labels_pos[:num_test_labels]
    test_labels_neg = train_labels_neg[:num_test_labels]

    train_labels_pos = train_labels_pos[num_test_labels:]
    train_labels_neg = train_labels_neg[num_test_labels:]

    return (train_labels_pos, test_labels_pos, 
            train_labels_neg, test_labels_neg, 
            user2movie, movie2user)


if __name__ == '__main__':
  # Test datasets
  ml_25m = ML25M()
  for train_label_batch in ml_25m.GetTrainEpoch(1):
    print(train_label_batch)
    break
