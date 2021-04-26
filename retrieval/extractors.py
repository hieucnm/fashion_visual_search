from __future__ import print_function, absolute_import

from retrieval.utils import flip_horizonal

import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F

from tqdm._tqdm import tqdm


class BaseExtractor(object):
    
    def __init__(self, model):
        super(BaseExtractor, self).__init__()
        self.model = model
        
    def _parse_data(self, inputs):
        return Variable(inputs['image'])
    
    def _forward_dataloader(self, data_loader):
        outputs = []
        for inputs in tqdm(data_loader):
            outputs.append(self._forward(self._parse_data(inputs)))
        outputs = torch.cat(outputs)
        return outputs
        
    def _forward(self, imgs):
        raise NotImplementedError

# ====================================================================================  

class EmbeddingExtractor(BaseExtractor):
    def _forward(self, imgs):
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(imgs).data.cpu()
            # flip_imgs = flip_horizonal(imgs)
            # flip_outputs = self.model(flip_imgs).data.cpu()
            # outputs = F.normalize(torch.cat([outputs, flip_outputs], dim=1))
        return outputs

# ====================================================================================  

class NormSoftmaxLogitExtractor(BaseExtractor):
    def _forward(self, embeddings):
        self.model.eval()
        with torch.no_grad():
            outputs = self.model._forward(embeddings).data.cpu()
        return outputs.softmax(dim=1)

    