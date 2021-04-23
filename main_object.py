
import warnings
warnings.filterwarnings('ignore')

# from IPython.core.display import display, HTML
# display(HTML("<style>.container { width:100% !important; }</style>"))
# display(HTML("<style>.output_png {display:table-cell; text-align:center; vertical-align:middle;}</style>"))


import os
import os.path as osp
import json
import time
import argparse
import numpy as np


import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

import torchvision
from pytorch_metric_learning import losses, samplers

import sys
sys.path.append(os.getcwd() + '/retrieval/')
import models, trainers, evaluators, datasets, utils
from utils.data import transforms as T
from utils.logging import Logger
from utils.meters import AverageMeter
from utils.serialization import load_checkpoint, save_checkpoint, read_json

torch.set_num_threads(4) # prevent cpu from exploding
__cwd__ = os.getcwd()

# =====================================================================

normalizer = T.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])

train_transformer = T.Compose([
    T.RectScale(160, 160),
    T.RandomSizedRectCrop(144, 144),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    normalizer,
])

test_transformer = T.Compose([
    T.RectScale(144, 144),
    T.ToTensor(),
    normalizer,
])

# =====================================================================

def get_criterion(args):

    n_class = max(json.load(open(args.class_path)).values()) + 1
    criterions = {
        'TripletMarginLoss' : losses.TripletMarginLoss(margin=args.margin),
        'ProxyNCALoss' : losses.ProxyNCALoss(n_class, args.embed_dim),
        'ProxyAnchorLoss' : losses.ProxyAnchorLoss(n_class, args.embed_dim, margin=args.margin),
        'NormalizedSoftmaxLoss' : losses.NormalizedSoftmaxLoss(args.temperature, args.embed_dim, n_class),
    }
    
    return criterions[args.loss_name]

# =====================================================================

def need_sampling(name):
    if name in ['ProxyNCALoss', 'ProxyAnchorLoss', 'NormalizedSoftmaxLoss']:
        return True
    return False

# =====================================================================


def test(args):
    
    start_time = time.time()
    print(f'Start time : {time.ctime()}')
    
    model = models.create('vsembedder', embed_dim=args.embed_dim, device=args.device)
    checkpoint = load_checkpoint(args.model_path)
    model.load(checkpoint)
    
    testset = datasets.TikiDataset(args.test_path, transform=test_transformer)
    testset.split_query_gallery(args.query_frac)
    test_loader  = DataLoader(testset, args.batch_size, shuffle=False)
    
    print('Testing ...')
    evaluator = evaluators.EmbeddingEvaluator(model)
    evaluator.evaluate(test_loader, query=testset.query, gallery=testset.gallery)
    
    print(f'Finish time : {time.ctime()}')
    print(f'Overall time : {(time.time() - start_time) / 60 :.2f} minutes')
    
    return


# =====================================================================

def train(args):
    
    start_time = time.time()
    sys.stdout = Logger(osp.join(args.log_dir, 'log.txt'))
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Start time : {time.ctime()}')
   
    print('Reading dataset ...')
    trainset = datasets.TikiDataset(args.train_path, transform=train_transformer)
    validset = datasets.TikiDataset(args.valid_path, transform=test_transformer)
    validset.split_query_gallery(args.query_frac)

    print('Defining everything ...')
    
    model = models.create('vsembedder', embed_dim=args.embed_dim, device=device)
    
    criterion = get_criterion(args).to(device)
    
    optimizer = optim.Adam(list(model.parameters()) + list(criterion.parameters()), 
                           lr=args.lr, weight_decay=args.weight_decay)
    
    
    sampler = None
    shuffle = True
    if need_sampling(args.loss_name):
        sampler = samplers.MPerClassSampler(trainset.targets, m=args.samples_per_class)
        shuffle = False
    
    train_loader = DataLoader(trainset, args.batch_size, num_workers=args.workers, shuffle=shuffle, sampler=sampler)
    valid_loader = DataLoader(validset, args.batch_size, num_workers=args.workers, shuffle=False)
    
    
    scheduler = MultiStepLR(optimizer, milestones=[20,30], gamma=0.1)
    trainer = trainers.EmbeddingTrainer(model, criterion)
    evaluator = evaluators.EmbeddingEvaluator(model)
    
    
    
    print(f'Start training! All arguments : {args}')
    print(f'Using device = {device} , sampler = {sampler}, scheduler = {scheduler}')
    best_result = 0
    
    for epoch in range(args.epochs):
        
        if epoch < args.warm_epochs:
            model.freeze_base()
        else:
            model.unfreeze_base()
        
        trainer.train(epoch, train_loader, optimizer, print_freq=args.print_freq)
        scheduler.step()

        if epoch < args.start_save:
            continue

        valid_result = evaluator.evaluate(valid_loader, query=validset.query, gallery=validset.gallery)

        is_best = valid_result > best_result
        best_result = max(valid_result, best_result)

        save_checkpoint(model.state_dict(), is_best, fpath=osp.join(args.log_dir, 'last_checkpoint_obj.pth'))

        print('\n * Finished epoch {:3d}  Top1 acc: {:5.1%}  Best: {:5.1%}{}\n'.
              format(epoch, valid_result, best_result, ' *' if is_best else ''))

    print('Finished!')
    print(f'Finish time : {time.ctime()}')
    print(f'Overall time : {(time.time() - start_time) / 3600 :.2f} hours')
    pass


if __name__ == '__main__':
    
    PREPROCESS_LOG_DIR = osp.join(__cwd__ , 'datasets/log_preprocessed')
    LOG_DIR = osp.join(__cwd__ , f'logs/{time.strftime("%Y%m%d-%H%M%S")}')
    
    
    parser = argparse.ArgumentParser(description="ImageSimilarity")

    # paths
    parser.add_argument('--train-path', type=str, default=osp.join(PREPROCESS_LOG_DIR, 'splits/train_valid_object/train.csv'))
    parser.add_argument('--valid-path', type=str, default=osp.join(PREPROCESS_LOG_DIR, 'splits/train_valid_object/valid.csv'))
    parser.add_argument('--class-path', type=str, default= osp.join(PREPROCESS_LOG_DIR, 'mappers/obj_mapper.json'))
                        
    parser.add_argument('--log-dir', type=str, default=LOG_DIR)
                        
    parser.add_argument('--model-path', type=str, default=osp.join(__cwd__, 'logs/20200528-002936/best_checkpoint_obj.pth'))
    parser.add_argument('--test-path' , type=str, default=osp.join(PREPROCESS_LOG_DIR, 'splits/train_valid_object/valid.csv'))
    

    # model
    parser.add_argument('--embed-dim', type=int, default=256)
    parser.add_argument('--device', type=str, default='cuda:3')
    
    # loss
    parser.add_argument('--margin', type=float, default=1.0)
    parser.add_argument('--temperature', type=float, default=0.05)
    parser.add_argument('--need-sampling', action='store_true')
    parser.add_argument('--samples_per_class', type=int, default=2)
    parser.add_argument('--loss-name', type=str, default='NormalizedSoftmaxLoss')

    # trainer
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('-b', '--batch-size', type=int, default=64)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--warm-epochs', type=int, default=10)
    parser.add_argument('--start-save', type=int, default=0)
    parser.add_argument('--print-freq', type=int, default=16)
    
    
    # evaluator
    parser.add_argument('--query-frac', type=float, default=0.2)
    
    
    #misc
    parser.add_argument('--to-do', type=str, choices=['train', 'test'], default='test')
    parser.add_argument('--seed', type=int, default=2020)

    args = parser.parse_args()
    
    if args.to_do == 'train':
        print('To do : Train')
        train(args)
    else:
        print('To do : Test')
        test(args)
