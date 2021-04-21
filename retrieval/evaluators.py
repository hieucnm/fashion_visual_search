from __future__ import print_function, absolute_import
import time
from collections import OrderedDict

import os
import sys
sys.path.append('/')

from evaluation_metrics import *
from utils.meters import AverageMeter
from utils import flip_horizonal, show_result

import numpy as np
from sklearn.metrics import *
import torch
from torch.autograd import Variable
import torch.nn.functional as F

# from pytorch_metric_learning.utils import AccuracyCalculator

       
# ====================================================================================

# ====================================================================================

class BaseEvaluator(object):
    
    def __init__(self, model):
        super(BaseEvaluator, self).__init__()
        self.model = model
    
    def evaluate(self, data_loader, **kwords):
        raise NotImplementedError
    
    def _parse_data(self, inputs):
        raise NotImplementedError
        
    def _forward_dataloader(self, data_loader):
        raise NotImplementedError
        
    def _forward(self, imgs):
        raise NotImplementedError

# ====================================================================================

class EmbeddingEvaluator(BaseEvaluator):
    
    def _parse_data(self, inputs):
        imgs = Variable(inputs['image'])
        item_ids = Variable(inputs['item_id'])
        return imgs, item_ids
        
    
    def _forward_dataloader(self, data_loader):
        
        self.model.eval()
        embeddings = OrderedDict()
        targets = OrderedDict()

        with torch.no_grad():
            for inputs in data_loader:
                imgs, item_ids = self._parse_data(inputs)
                outputs = self._forward(imgs)

                for fname, output, item_id in zip(inputs['fname'], outputs, item_ids.tolist()):
                    embeddings[fname] = output
                    targets[fname] = item_id

        return embeddings, targets
    
    
    def _forward(self, imgs):
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(imgs).data.cpu()
            flip_imgs = flip_horizonal(imgs)
            flip_outputs = self.model(flip_imgs).data.cpu()
            outputs = F.normalize(torch.cat([outputs, flip_outputs], dim=1))
        return outputs


    def extract_query_gallery(self, embeddings, query, gallery):
        if len(query) == 0:
            print('No query to get embeddings! Only get embeddings of gallery images')
            query_embeds = []
            query_ids = []
        else:
            query_embeds = torch.stack([embeddings[f] for f, _ in query]).numpy()
            query_ids = np.array([iid for _, iid in query])

        gallery_embeds = torch.stack([embeddings[f] for f, _ in gallery]).numpy()
        gallery_ids = np.array([iid for _, iid in gallery])

        return query_embeds, gallery_embeds, query_ids, gallery_ids

    
    def evaluate(self, data_loader, **kwords):
        """
            data_loader: batch generator of validset
            query_ids: a list of (fnames, iid), maybe 20% part of validset that be used to query
            gallery_ids: a list of (fnames, iid), 80% remain of validset that be used to find best match
        
        """
        query = kwords['query']
        gallery = kwords['gallery']
        
        embeddings, _ = self._forward_dataloader(data_loader)
        query_embeds, gallery_embeds, query_ids, gallery_ids = self.extract_query_gallery(embeddings, query, gallery)
        
        distmat = cosine_similarity(query_embeds, gallery_embeds)
        mAP = mean_ap_v2(distmat, query_ids, gallery_ids, sort_asc=False)
        cmc_scores = topk_accuracy(distmat, query_ids, gallery_ids, sort_asc=False)
        show_result(mAP, cmc_scores)
        
        return cmc_scores[0]

    
    
# ====================================================================================

class ClassifyingEvaluator(BaseEvaluator):
    
    def _parse_data(self, inputs):
        imgs = Variable(inputs['image'])
        targets = Variable(inputs['category'])
        return imgs, targets
    
    
    def _forward_dataloader(self, data_loader):
        self.model.eval()
        predicts, targets = [], []

        with torch.no_grad():
            for inputs in data_loader:
                
                imgs, labels = self._parse_data(inputs)
                outputs = self._forward(imgs)
                labels  = labels.cpu().numpy()

                predicts.append(outputs)
                targets.append(labels)
        
        return np.concatenate(predicts), np.concatenate(targets)
        
    
    def _forward(self, imgs):
        with torch.no_grad():
            outputs = self.model(Variable(imgs)).softmax(dim=1).argmax(dim=1).cpu().numpy()
        return outputs
        
    
    def evaluate(self, data_loader):
        predicts, targets = self._forward_dataloader(data_loader)
        return accuracy_score(targets, predicts)

    
# ====================================================================================

class UncertaintyClassifyingEvaluator(ClassifyingEvaluator):
    
    def _forward_dataloader(self, data_loader):
        self.model.eval()
        predicts, targets = [], []

        with torch.no_grad():
            for inputs in data_loader:
                
                imgs, labels = self._parse_data(inputs)
                outputs, _ = self._forward(imgs)
                labels  = labels.cpu().numpy()

                predicts.append(outputs)
                targets.append(labels)
        
        return np.concatenate(predicts), np.concatenate(targets)

    
    def _forward(self, imgs):
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(Variable(imgs))
            uncertainties = outputs[:,-1].pow(2).cpu().numpy()
            outputs = outputs[:,:-1].softmax(dim=1).argmax(dim=1).cpu().numpy()
        return outputs, uncertainties
