from __future__ import absolute_import

import numpy as np
import torch
import sklearn.metrics.pairwise as metrics

def euclidean_distance(x, y):
    return torch.Tensor(metrics.euclidean_distances(x,y))

def jaccard_similarity(x, y):
    m, n = len(x), len(y)
    return torch.cat([np.logical_xor(u,y).float().mean(1) for u in x]).view(m, n)

def jaccard_distance(x, y):
    return 1 - jaccard_similarity(x, y)


def cosine_similarity(x, y):
    return torch.Tensor(metrics.cosine_similarity(x,y))

def cosine_distance(x, y):
    return 1 - cosine_similarity(x,y)