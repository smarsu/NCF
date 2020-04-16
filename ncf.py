# Copyright (c) 2020 smarsu. All Rights Reserved.

"""NCF net.

Reference:
https://github.com/NVIDIA/DeepLearningExamples/blob/master/TensorFlow/Recommendation/NCF
"""

import numpy as np
import torch
from torch.nn.parameter import Parameter


class NCF(torch.nn.Module):
  """Construct a pytorch NCF model."""
  def __init__(self, num_users, num_items, mf_dim, mlp_layer_sizes, device):
    """
    Args:
      num_users: int. The total num of users.
      num_items: int. The total num of items.
      mf_dim: int. The input dim of mf layer.
      mlp_layer_sizes: List of int. The hidden size of each mlp layers.
    """
    super(NCF, self).__init__()

    print('num_users ... {}'.format(num_users))
    print('num_items ... {}'.format(num_items))

    self.user_mf_embd = Parameter(torch.Tensor(num_users, mf_dim))
    torch.nn.init.kaiming_normal_(self.user_mf_embd.data, nonlinearity='relu')
    self.item_mf_emdb = Parameter(torch.Tensor(num_items, mf_dim))
    torch.nn.init.kaiming_normal_(self.item_mf_emdb.data, nonlinearity='relu')

    assert mlp_layer_sizes[0] % 2 == 0, 'Expected mlp_layer_sizes[0] % 2 == 0,' \
      ' get {}'.format(mlp_layer_sizes[0])
    self.user_mlp_embd = Parameter(
      torch.Tensor(num_users, mlp_layer_sizes[0] // 2))
    torch.nn.init.kaiming_normal_(self.user_mlp_embd.data, nonlinearity='relu')
    self.item_mlp_emdb = Parameter(
      torch.Tensor(num_items, mlp_layer_sizes[0] // 2))
    torch.nn.init.kaiming_normal_(self.item_mlp_emdb.data, nonlinearity='relu')

    num_mlp_layers = len(mlp_layer_sizes) - 1 

    self.mlp = torch.nn.Sequential(*[torch.nn.Sequential(
                                      torch.nn.Linear(
                                        mlp_layer_sizes[layer], 
                                        mlp_layer_sizes[layer + 1], 
                                        bias=True),
                                      torch.nn.ReLU(),
                                      torch.nn.Dropout(0.5))
                                    for layer in range(num_mlp_layers)])

    self.linear = torch.nn.Linear(
      mf_dim + mlp_layer_sizes[num_mlp_layers], 
      1, 
      bias=True)  # TODO(smarsu): Convert bias to False

    self.bce_loss = torch.nn.BCELoss(reduction='mean')


  def forward(self, users, items):
    """
    Args:
      users: torch tensor. shape: [batch_size]. The users id.
      items: torch tensor. shape: [batch_size]. The items id.

    Returns:
      logits: torch tensor. shape [batch_size]. The output of last linear layer.
    """
    # Pay attention to the difference between torch.gather and torch.index_select
    # https://pytorch.org/docs/stable/torch.html?highlight=gather#torch.gather
    xmf_user = torch.index_select(self.user_mf_embd, 0, users)
    xmf_item = torch.index_select(self.item_mf_emdb, 0, items)
    xmf = xmf_user * xmf_item

    xmlp_user = torch.index_select(self.user_mlp_embd, 0, users)
    xmlp_item = torch.index_select(self.item_mlp_emdb, 0, items)
    xmlp = self.mlp(torch.cat([xmlp_user, xmlp_item], 1))

    logits = torch.cat([xmf, xmlp], 1)
    logits = self.linear(logits)
    return logits


  def ncf_loss(self, logits, labels):
    """Compute the ncf loss.
    
    Args:
      logits: Logits without sigmoid.
      labels:

    Return:
      loss:
    """
    return self.bce_loss(torch.sigmoid(logits), labels)


if __name__ == '__main__':
  # Test NCF model
  ncf = NCF(num_users=10, 
            num_items=10, 
            mf_dim=10, 
            mlp_layer_sizes=[256, 256, 128, 64])
  users = torch.tensor([0])
  items = torch.tensor([0])
  labels = torch.tensor([1], dtype=torch.float32)
  logits = ncf(users, items)
  loss = ncf.ncf_loss(logits, labels)
  print(logits)
  print(loss)
