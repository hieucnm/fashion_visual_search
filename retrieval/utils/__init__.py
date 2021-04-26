from __future__ import absolute_import

import torch
import numpy as np


def to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.cpu().numpy()
    elif type(tensor).__module__ != 'numpy':
        raise ValueError("Cannot convert {} to numpy array"
                         .format(type(tensor)))
    return tensor


def to_torch(ndarray):
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor"
                         .format(type(ndarray)))
    return ndarray

def is_trainable(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params > 0



def flip_horizonal(imgs):
    inv_idx   = torch.arange(imgs.size(3) - 1, -1, -1).long()  # N x C x H x W
    flip_imgs = imgs.index_select(3, inv_idx)
    return flip_imgs

def trim_image(img, pixel_max=250):
    isnot_white = (img <= pixel_max)[:,:,0]
    trimmed_img = img[:,np.any(isnot_white, axis=0),:][np.any(isnot_white, axis=1),:,:]
    return trimmed_img


def show_result(mAP, cmc_scores):
    print(f'Top1  Accuracy : {cmc_scores[0] * 100 :.2f}%')
    print(f'Top5  Accuracy : {cmc_scores[4] * 100 :.2f}%')
    print(f'Top10 Accuracy : {cmc_scores[9] * 100 :.2f}%')
    print(f'mAP@1 : {mAP * 100 :.2f}%')
    pass


def restrict_ordinate(x, max_x):
    x = max(int(x),0)
    x = min(int(x), max_x)
    return x

def restrict_bbox(x1,x2,y1,y2,max_x,max_y):
    x1 = restrict_ordinate(x1, max_x)
    x2 = restrict_ordinate(x2, max_x)
    y1 = restrict_ordinate(y1, max_y)
    y2 = restrict_ordinate(y2, max_y)
    return x1,x2,y1,y2
