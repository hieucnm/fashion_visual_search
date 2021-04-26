from __future__ import absolute_import
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
__cwd__ = os.getcwd()


import warnings
warnings.filterwarnings('ignore')
# ---------------------------------------------

import os.path as osp
import argparse

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

import torch
from torchvision import transforms
from pretrainedmodels.utils import ToRange255
from pretrainedmodels.utils import ToSpaceBGR

import sys
sys.path.append(__cwd__)
sys.path.append(__cwd__ + '/detector')
# sys.path.append(__cwd__ + '/retrieval')
# sys.path.append(__cwd__ + '/demo')
# sys.path.append(__cwd__ + '/classification_metric_learning')


from detector.yolo.utils.utils import load_classes
from detector.predictors.YOLOv3 import YOLOv3Predictor
from retrieval.utils.serialization import load_checkpoint
import classification_metric_learning.metric_learning.modules.featurizer as featurizer

from demo.demo_detector import DemoDetector
from demo.demo_backend import DemoBackend
from demo.demo_frontend import DemoFrontend
from demo.demo_controller import DemoController


torch.set_num_threads(2) # prevent cpu from exploding
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATABASE_DIR = osp.join(__cwd__, 'datasets/database')

# ==========================================================================================================
parser = argparse.ArgumentParser(description="ImageSimilarity")
parser.add_argument('--item-db-json-path' , type=str, default=osp.join(DATABASE_DIR, 'item_database.json'))
parser.add_argument('--obj-db-json-path'  , type=str, default=osp.join(DATABASE_DIR, 'object_database.json'))

parser.add_argument('--item-cluster-path' , type=str, default=osp.join(DATABASE_DIR, 'item_kmean_clustering.pkl'))
parser.add_argument('--obj-cluster-path'  , type=str, default=osp.join(DATABASE_DIR, 'object_kmean_clustering.pkl'))

parser.add_argument('--embedder-path', type=str, metavar='PATH', 
               default=osp.join(__cwd__ , 'classification_metric_learning/output/InShop/2048/resnet50_75/epoch_30.pth'))


parser.add_argument('--embed-dim', type=int, default=2048)
args = parser.parse_args([])


# ==========================================================================================================


def load_all():
    print('Loading embedder model ...')
    model_factory = getattr(featurizer, 'resnet50')
    embedder = model_factory(args.embed_dim, device)
    embedder.load_state_dict(load_checkpoint(args.embedder_path))
    embedder = embedder.to(device)
    
    
    test_transformer = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(max(embedder.input_size)),
        transforms.ToTensor(),
        ToSpaceBGR(embedder.input_space == 'BGR'),
        ToRange255(max(embedder.input_range) == 255),
        transforms.Normalize(mean=embedder.mean, std=embedder.std)
    ])
    
    
    print('Loading detector model ...')
    yolo_params = {
        "model_def" : "detector/yolo/df2cfg/yolov3-df2.cfg",
        "weights_path" : "detector/yolo/weights/yolov3-df2_15000.weights",
        "class_path":"detector/yolo/df2cfg/df2.names",
        "conf_thres" : 0.5,
        "nms_thres" :0.4,
        "img_size" : 416,
        "device" : device
    }
    
    detectron = YOLOv3Predictor(params=yolo_params)
    
    position_dict = {
        'short-sleeve-top' : 'is_upper',
        'long-sleeve-top' : 'is_upper',
        'long-sleeve-outwear' : 'is_upper', 
        'short-sleeve-outwear' : 'is_upper', 
        'long-sleeve-dress' : 'is_upper', 
        'short-sleeve-dress' : 'is_upper', 
        'sling-dress' : 'is_upper', 
        'vest-dress' : 'is_upper', 
        'vest' : 'is_upper', 
        'sling' : 'is_upper',

        'trousers' : 'is_lower',
        'shorts' : 'is_lower',
        'skirt' : 'is_lower',
    }

    classes = load_classes(yolo_params["class_path"])
    cmap = plt.get_cmap("rainbow")
    colors = np.array([cmap(i) for i in np.linspace(0, 1, 13)])
    demo_detector = DemoDetector(detectron, classes, colors, position_dict)
    
    
    print('Initializing demo modules ...')
    backend_params = {
        'device' : device,
        'transform' : test_transformer,
        'embedder' : embedder,
        'demo_detector' : demo_detector,

        'item_db' : pd.read_json(args.item_db_json_path),
        'obj_db' : pd.read_json(args.obj_db_json_path), 

        'item_cluster' : pickle.load(open(args.item_cluster_path, 'rb')),
        'obj_cluster' : pickle.load(open(args.obj_cluster_path, 'rb')), 

    }
    
    demo_backend = DemoBackend(backend_params)
    demo_frontend = DemoFrontend(demo_dir=osp.join(__cwd__, 'datasets/images_demo'))
    demo = DemoController(frontend=demo_frontend, backend=demo_backend)
    print('Demo initialized!')
    
    return demo