import warnings
warnings.filterwarnings('ignore')

import torch
from PIL import Image
import pandas as pd

from retrieval.evaluation_metrics import cosine_distance
# ====================================================================================================


class DemoBackend(object):
    
    def __init__(self, params):
        
        self.transform = params['transform']
        self.device = params['device']
        self.embedder  = params['embedder']
        self.demo_detector = params['demo_detector']
        self.item_database = params['item_db']
        self.obj_database  = params['obj_db']
        self.item_cluster  = params['item_cluster']
        self.obj_cluster   = params['obj_cluster']
        
        self.topk_clusters = 3

    
    def search(self, query_embed, gallery, topk=20):
        
        gallery_embeds = torch.Tensor(gallery.embed)
        result = gallery.drop(columns=['embed'])
        result['score'] = cosine_distance(query_embed, gallery_embeds).numpy().ravel()
        result = result.sort_values(by='score', ascending=True).reset_index(drop=True)
        result['rank'] = result.index + 1
        result = result.drop_duplicates('item_id').reset_index(drop=True)
        return result.head(topk)
    
    
    def run(self, image_path, topk=20):
        img = Image.open(image_path).convert('RGB')
        img = self.transform(img).unsqueeze(0).to(self.device)
        query_embed = self.forward_img(img)
        query_clusters = self.item_cluster.transform(query_embed.numpy()).ravel().argsort()[:self.topk_clusters]
        
        item_gallery = self.item_database[self.item_database.cluster.isin(query_clusters)].reset_index(drop=True)
        item_most_similar = self.search(query_embed, item_gallery, topk)
        
        detected_objects = self.demo_detector.run(image_path)
        objects_most_similar = []
        if len(detected_objects) != 0:
            for obj_classname, render_img, cut_bbox in detected_objects:
                obj_bbox = self.transform(Image.fromarray(cut_bbox)).unsqueeze(0).to(self.device)
                obj_embed = self.forward_img(obj_bbox)
                
                obj_clusters = self.obj_cluster.transform(obj_embed.numpy()).ravel().argsort()[:self.topk_clusters]
                # obj_gallery = self.obj_database
                obj_gallery = self.obj_database[self.obj_database.cluster.isin(obj_clusters)].reset_index(drop=True)
                obj_gallery = obj_gallery[obj_gallery[obj_classname]].reset_index(drop=True)
                
                obj_most_similar = self.search(obj_embed, obj_gallery, topk)
                objects_most_similar.append((obj_classname, render_img, obj_most_similar))
        
        
        return item_most_similar, objects_most_similar
    
    
    def forward_img(self, img):
        self.embedder.eval()
        with torch.no_grad():
            return self.embedder(img).detach().cpu()
