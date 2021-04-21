from __future__ import print_function, absolute_import
import time

import torch
from torch.autograd import Variable
from utils.meters import AverageMeter
from sklearn.metrics import accuracy_score
from pytorch_metric_learning import losses
# from pytorch_metric_learning.utils import AccuracyCalculator
from pytorch_metric_learning import miners, losses

import sys
sys.path.append('/')
from evaluation_metrics import topk_accuracy, cosine_similarity, euclidean_distances

# ==================================================


class BaseTrainer(object):
    
    def __init__(self, model, criterion):
        super(BaseTrainer, self).__init__()
        
        self.losses_zoo = ['TripletMarginLoss', 'ProxyAnchorLoss', 
                           'ProxyNCALoss', 'NormalizedSoftmaxLoss', 
                           'CrossEntropyLoss']
        
        self.need_mining_losses = ['TripletMarginLoss', 'ProxyAnchorLoss']
        
        self.model = model
        self.criterion = criterion
        self.miner = None
        self.verify()
    
    
    def verify(self):
        assert self.criterion.__class__.__name__ in self.losses_zoo, f'Unsupported loss: {self.criterion}'
        
        if self.criterion.__class__.__name__ in self.need_mining_losses:
            self.miner = miners.MultiSimilarityMiner(epsilon=0.1)
        return
        

    def train(self, epoch, data_loader, optimizer, print_freq=1):
        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        accuracies = AverageMeter()

        end = time.time()
        
        for i, inputs in enumerate(data_loader):
            
            data_time.update(time.time() - end)
            inputs, targets = self._parse_data(inputs)
            
            optimizer.zero_grad()
            loss, acc = self._forward(inputs, targets)
            
            loss.backward()
            optimizer.step()

            losses.update(loss.item(), targets.size(0))
            accuracies.update(acc, targets.size(0))
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'Acc {:.2%} ({:.2%})\t'
                      .format(epoch, i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,
                              accuracies.val, accuracies.avg))
    

    def _parse_data(self, inputs):
        raise NotImplementedError

    def _forward(self, inputs, targets):
        raise NotImplementedError
        
    def _calc_inner_batch_accuracy(self, outputs, targets):
        raise NotImplementedError


# ==================================================


class EmbeddingTrainer(BaseTrainer):
    
    def _parse_data(self, inputs):
        imgs = Variable(inputs['image'])
        targets = Variable(inputs['item_id']).to(self.model.device)
        return imgs, targets
    
    def _calc_inner_batch_accuracy(self, outputs, targets):
        outputs = outputs.detach().cpu().numpy()
        targets = targets.detach().cpu().numpy()
        distmat = cosine_similarity(outputs, outputs)
        return topk_accuracy(distmat, targets, targets, topk=1)[0]

    def _forward(self, inputs, targets):
        outputs = self.model(inputs)
        
        if self.miner is not None:
            hard_points = self.miner(outputs, targets)
            loss = self.criterion(outputs, targets, hard_points)
        else:
            loss = self.criterion(outputs, targets)
            
        acc = self._calc_inner_batch_accuracy(outputs, targets)

        return loss, acc


# ==================================================


class ClassifyingTrainer(BaseTrainer):
    
    def _parse_data(self, inputs):
        imgs = Variable(inputs['image'])
        targets = Variable(inputs['category']).to(self.model.device)
        return imgs, targets
    
    def _calc_inner_batch_accuracy(self, outputs, targets):
        predicts = outputs.softmax(dim=1).argmax(dim=1).detach().cpu().numpy()
        targets = targets.detach().cpu().numpy()
        return accuracy_score(targets, predicts)

    def _forward(self, inputs, targets):
        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)
        acc = self._calc_inner_batch_accuracy(outputs, targets)
        return loss, acc


# ==================================================


class UncertaintyClassifyingTrainer(ClassifyingTrainer):

    def _forward(self, inputs, targets):
        outputs = self.model(inputs)
        
        # ignore last element of output softmax layer
        # because it is the uncertainty scale
        # scale output by the uncertainty sigma^2
        outputs = outputs[:,:-1] / outputs[:,-1].pow(2).unsqueeze(1)
        
        loss = self.criterion(outputs, targets)
        acc = self._calc_inner_batch_accuracy(outputs, targets)
        return loss, acc
