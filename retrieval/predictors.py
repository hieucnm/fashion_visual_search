from __future__ import print_function, absolute_import

# import os
# import sys
# import os.path as osp
# sys.path.append(osp.dirname(os.getcwd()))
# sys.path.append(osp.dirname(osp.dirname(os.getcwd())))


from retrieval.datasets import TikiDataset, TikiEmbeddingDataset, AutomaticTestDataset
from retrieval.extractors import EmbeddingExtractor, NormSoftmaxLogitExtractor
# from common.image_processing import refactor_value_range

import torch
from pretrainedmodels.utils import ToRange255, ToSpaceBGR
from torch.utils.data import DataLoader
from torchvision import transforms

import classification_metric_learning.metric_learning.modules.featurizer as featurizer
from classification_metric_learning.metric_learning.modules.losses import NormSoftmaxLoss
from classification_metric_learning.evaluation.retrieval import _retrieve_knn_faiss_gpu_inner_product

import numpy as np
import pandas as pd
import matplotlib.image as mpimg
from PIL import Image


dim = 256
num_workers = 4
batch_size = 64
model_name = 'resnet50'
train_num_instance = 11201
device = torch.device('cuda')


eval_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224), # max(model.input_size)
    transforms.ToTensor(),
    ToSpaceBGR(False), # model.input_space == 'BGR'
    ToRange255(False), # max(model.input_range) == 255
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # model.mean, model.std
])


class Predictor(object):
    
    def __init__(self, db_path, db_embed_path, db_class_path, model_path, clf_path, 
                         train_num_instance=train_num_instance, 
                         transform=eval_transform, 
                         model_name=model_name, 
                         embed_dim=dim, 
                         device=device):
        
        self.db_path = db_path
        self.db_embed_path = db_embed_path
        self.db_class_path = db_class_path
        self.model_path = model_path
        self.clf_path = clf_path
        
        self.train_num_instance = train_num_instance
        self.transform = eval_transform
        self.model_name = model_name
        self.dim = embed_dim
        self.device = device
        
        self.model = self._load_model()
        self.clf = self._load_clf()
        self.database = TikiDataset(db_path, self.transform)
        self.db_embeddings = np.load(self.db_embed_path)
        self.db_classes = np.load(self.db_class_path) 
        self.embedding_extractor = EmbeddingExtractor(self.model)
        self.logit_extractor = NormSoftmaxLogitExtractor(self.clf)


    def _load_model(self):
        model_factory = getattr(featurizer, self.model_name)
        model = model_factory(self.dim, self.device)
        model.load_state_dict(torch.load(self.model_path))
        return model
    
    def _load_clf(self):
        clf = NormSoftmaxLoss(self.dim, self.train_num_instance, device=self.device)
        clf.load_state_dict(torch.load(self.clf_path))
        return clf
    
    def _extract_features(self, imgs):
        if isinstance(imgs, DataLoader):
            embeddings = self.embedding_extractor._forward_dataloader(imgs)
            logit_db = TikiEmbeddingDataset(features=embeddings.numpy())
            logit_dataloader = DataLoader(logit_db, batch_size=batch_size, num_workers=num_workers)
            logits = self.logit_extractor._forward_dataloader(logit_dataloader)    
        else:
            embeddings = self.embedding_extractor._forward(imgs)
            logits = self.logit_extractor._forward(embeddings)
        embeddings = embeddings.numpy()
        classes = logits.argsort(dim=1, descending=True)[:,:5].numpy()
        return embeddings, classes
    
    
    def search(self, imgs_as_nparray, topk=5, distinct_uid=True):
        if not isinstance(imgs_as_nparray, list):
            imgs_as_nparray = [imgs_as_nparray]
        
        # imgs = refactor_value_range(imgs_as_nparray, range_type='uint')
        imgs = torch.stack([self.transform(Image.fromarray(img).convert('RGB')) for img in imgs])
        dists, indices, classes = self._forward(imgs, topk=100)
        
        results = []
        for i in range(imgs.shape[0]):
            seen = []
            nearests = []
            for score, idx in zip(dists[i], indices[i]):
                sample = self.database.data.iloc[idx]
                n = sample.path.split('_')[-1].split('.')[0]
                path = sample.path.replace(f'_{n}', '') # convert object_path to image_path
                item_id = '_'.join([sample.item_id, n])
                if path in seen:
                    continue
                nearests.append({
                    'image' : mpimg.imread(path),
                    'item_id' : sample.item_id,
                    'score' : score,
                    'class' : self.db_classes[idx].tolist()[:topk],
                    'is_box' : n != '',
                })
                seen.append(path)
                if len(seen) == topk:
                    break
            results.append({'nearests' : nearests, 'class' : classes[i]})
        return results
    

    def _forward(self, imgs, topk=5):
        embeddings, classes = self._extract_features(imgs)
        dists, indices = _retrieve_knn_faiss_gpu_inner_product(embeddings, self.db_embeddings, topk)
        return dists, indices, classes
    
    
    def test(self, image_paths, topk=5):
        
        testset = AutomaticTestDataset(image_paths, self.transform)
        testloader = DataLoader(testset, batch_size=batch_size, num_workers=num_workers)
        _, indices, classes = self._forward(testloader, topk)
        nearest_classes = np.array([[self.db_classes[i][:topk] for i in row] for row in indices])
        
        # classes.shape (2D) = num_queries, self.db_classes.shape[1]
        # nearest_classes.shape (3D) = num_queries, topk, self.db_classes.shape[1]
        # topk_classes.shape (2D) = num_queries, k x self.db_classes.shape[1]
        
        acc = {}
        for k in range(1, topk+1):
            topk_classes = nearest_classes[:,:k,:].reshape(len(image_paths), -1)
            acc[f'R@{k}'] = [not set(qc).isdisjoint(topk_classes[i]) for i,qc in enumerate(classes)]
        return acc
