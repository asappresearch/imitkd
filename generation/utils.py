from typing import List

import numpy as np
import torch
from torch import Tensor


def topk_2d(x: Tensor, k: int):
    '''
    Batch topk function applied to the last two dimensions.
    '''
    last_dim = x.size(-1)
    x_ = torch.flatten(x, start_dim=-2, end_dim=-1)
    values, flat_indices = torch.topk(x_, k=k, dim=-1, sorted=True)
    indices1 = flat_indices // last_dim
    indices2 = flat_indices % last_dim
    return values, indices1, indices2

def nan_mean(x: Tensor, dim: int = 0):
    y = x.clone()
    mask = ~torch.isnan(y)
    counts = mask.byte().sum(dim=dim).float()
    y[~mask] = 0
    sums = y.sum(dim=dim)
    return sums / counts

def pad_to_len(x: Tensor, len: int, pad_idx: int, dim: int = 0):
    gap = len - x.size(dim)
    if gap > 0:
        pad_shape = list(x.size())
        pad_shape[dim] = gap
        pads = torch.tensor(pad_idx).repeat(pad_shape)
        return torch.cat([x, pads], dim=dim)
    else:
        return x

def pad_and_cat(lst: List[Tensor], pad_idx: int, dim: int = 0):
    n_dim = len(lst[0].size())
    if not all([len(x.size()) == n_dim for x in lst]):
        raise ValueError('All tensors must have same number of dimensions!')
    for i in range(n_dim):
        if i != dim:
            max_len = np.max([x.size(i) for x in lst])
            for j in range(len(lst)):
                lst[j] = pad_to_len(lst[j], max_len, pad_idx, dim=i)
    return torch.cat(lst, dim)

def tensor_to_list(data: Tensor, pad_idx: int):
    '''
    data: 2-dimensional Tensor
    '''
    lst = []
    for x in data:
        length = torch.sum(x != pad_idx).item()
        lst.append(x[:length])
    return lst

def delete_extra_pads(data: Tensor, pad_idx: int):
    '''
    data: 2-dimensional Tensor
    '''
    max_len = (data != pad_idx).sum(dim=1).max().item()
    return data[:, :max_len]
