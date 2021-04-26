from __future__ import absolute_import
from collections import defaultdict

import numpy as np
import torch
from sklearn.metrics import average_precision_score

from retrieval.utils import to_numpy, to_torch
from retrieval.evaluation_metrics import *

VERY_LARGE_DISTANCE = 1e6
VERY_SMALL_SIMILARITY = -1

MIN_DISTANCE = 0
MAX_SIMILARITY = 1

def ensure_numpy_array(distmat, x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    distmat = to_numpy(distmat)
    m, n = distmat.shape
    assert m == x.size, "num of queries and num rows of distance matrix are not the same"
    assert n == y.size, "num of galleries and num columns of distance matrix are not the same"
    return distmat, x, y


def ensure_torch_tensor(distmat, x, y):
    x = to_torch(x)
    y = to_torch(y)
    distmat = to_torch(distmat)
    m, n = distmat.shape
    assert (x.shape[0],y.shape[0]) == distmat.shape, "num of queries and galleries does not match distance matrix shape"
    return distmat, x, y


def sort_distmat(distmat, sort_asc):
    
    """
        Sort and find correct matches
        By firstly, we should ignore where the distance = 0, because they are from the same images
        If we use distance, e.g euclidean, then sort_asc = False and we replace 0 by a very large value
        In contrast, if we use similarity, e.g cosine, then sort_asc = True and we replace 1 by a very small value.
    """
    distmat = to_torch(distmat)
    
    if sort_asc:
        distmat = torch.where(distmat == 0, torch.ones_like(distmat)*VERY_LARGE_DISTANCE, distmat)
    else:
        distmat = torch.where(distmat == 1, torch.ones_like(distmat)*VERY_SMALL_SIMILARITY, distmat)

    distmat, indices = distmat.sort(dim=1, descending = not sort_asc)
    
    return distmat, indices
    

def topk_accuracy(distmat, query_ids, gallery_ids, topk=20, sort_asc=False):
    
    # Ensure numpy array
    distmat, query_ids, gallery_ids = ensure_torch_tensor(distmat, query_ids, gallery_ids)
    
    # Sort and find correct matches
    _, indices = sort_distmat(distmat, sort_asc)
    matches = (gallery_ids[indices] == query_ids[:, np.newaxis]).float()[:,:topk]
    
    match_indices = np.where(matches)
    
    if len(match_indices[1]) == 0:
        print('No valid query to calculate CMC!')
        return [0]*topk
    
    # for each row, assign True for all position from first seen True position
    matches = np.array([[False]*i_col + [True]*(matches[i_row].shape[0] - i_col)
                        for i_row,i_col in zip(range(len(matches)), match_indices[1])])
    
    return matches.sum(axis=0) / len(query_ids)


def mean_ap_v2(distmat, query_ids, gallery_ids, sort_asc=False):
    # Ensure numpy array
    distmat, query_ids, gallery_ids = ensure_torch_tensor(distmat, query_ids, gallery_ids)
    
    # Sort and find correct matches
    indices = distmat.argsort(dim=1, descending = not sort_asc)
    matches = (gallery_ids[indices] == query_ids[:, np.newaxis]).float()
    
    APs = []
    for row,dist in zip(matches, distmat):
        # similarly to topk_accuracy, we should remove same image with query
        row = row[dist != MIN_DISTANCE if sort_asc else dist != MAX_SIMILARITY]
        
        match_indices = np.where(row)[0]
        if len(match_indices) != 0:
            APs.append(np.mean([row[:idx+1].mean() for idx in match_indices]))
    
    if len(APs) == 0:
        print('No valid query to calculate mAP!')
        return 0
    return np.mean(APs)
        


def cmc(distmat, query_ids, gallery_ids, topk=20, first_match_break=True):
    
    # Ensure numpy array
    query_ids = np.asarray(query_ids)
    gallery_ids = np.asarray(gallery_ids)
    distmat = to_numpy(distmat)
    m, n = distmat.shape
    
    # Sort and find correct matches
    indices = np.argsort(distmat, axis=1)
    matches = (gallery_ids[indices] == query_ids[:, np.newaxis])
    
    # Compute CMC for each query
    ret = np.zeros(topk)
    num_valid_queries = 0
    for i in range(m):
        
        valid = gallery_ids[indices[i]] != query_ids[i]
        if not np.any(matches[i, valid]): continue
        
        index = np.nonzero(matches[i, valid])[0]
        delta = 1. / len(index)
        
        for j, k in enumerate(index):
            if k - j >= topk: break
            if first_match_break:
                ret[k - j] += 1
                break
            ret[k - j] += delta
                
        num_valid_queries += 1

    if num_valid_queries == 0:
        # raise RuntimeError("No valid query")
        print("No valid query")
        return [0]*topk
    return ret.cumsum() / num_valid_queries


def mean_ap(distmat, query_ids, gallery_ids):
    
    # Ensure numpy array
    query_ids = np.asarray(query_ids)
    gallery_ids = np.asarray(gallery_ids)
    distmat = to_numpy(distmat)
    m, n = distmat.shape
    
    # Sort and find correct matches
    indices = np.argsort(distmat, axis=1)
    matches = (gallery_ids[indices] == query_ids[:, np.newaxis])
    
    # Compute AP for each query
    aps = []
    for i in range(m):
        # Filter out the same id and same camera
        valid = gallery_ids[indices[i]] != query_ids[i]
        y_true = matches[i, valid]
        y_score = -distmat[i][indices[i]][valid]
        if not np.any(y_true): continue
        aps.append(average_precision_score(y_true, y_score))
    
    if len(aps) == 0:
        # raise RuntimeError("No valid query")
        return 0
    return np.mean(aps)



def _unique_sample(ids_dict, num):
    mask = np.zeros(num, dtype=np.bool)
    for _, indices in ids_dict.items():
        i = np.random.choice(indices)
        mask[i] = True
    return mask