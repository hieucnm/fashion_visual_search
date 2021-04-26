
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset
from PIL import Image

class TikiDataset(Dataset):
    
    def __init__(self, csv_path, transform=None):
        self.csv_path = csv_path
        self.transform = transform
        self.data = pd.read_csv(self.csv_path)
        
        # self.n_class = self.data.item_id.max() + 1
        self.targets = self.data.item_id
        self.query = []
        self.gallery = []
    
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img = Image.open(self.data.iloc[idx].path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        
        sample = {}
        
        sample['image'] = img
        sample['item_id'] = self.data.iloc[idx].item_id
        sample['fname'] = self.data.iloc[idx].fname
        sample['category'] = self.data.iloc[idx].category
        
        return sample
    
    
    def filter_data(self):
        # firstly, we should only use items having atleast 2 images
        if 'num_images' not in self.data.columns:
            num_images = self.data.groupby('item_id').fname.count().to_frame('num_images').reset_index()
            self.data = self.data.merge(num_images, on='item_id')
        
        self.data = self.data[self.data.num_images >= 2].reset_index(drop=True)
        return
    
    
    def split_query_gallery(self, query_frac=0.2):
        
        # for each item_id, use 1 image as query, and remain images as gallery
        query = self.data.drop_duplicates('item_id')[['fname', 'item_id']]
        self.query = [tuple(x) for x in query.values]
        
        query_fnames = set(query.fname)
        gallery = self.data[~self.data.fname.isin(query_fnames)][['fname', 'item_id']]
        self.gallery = [tuple(x) for x in gallery.values]
        pass
    
    # for testing
    def use_all_items_as_gallery(self):
        self.query = []
        self.gallery = [tuple(x) for x in self.data[['fname', 'item_id']].values]
        pass
    
    def use_all_items_as_query_gallery(self):
        self.query = [tuple(x) for x in self.data[['fname', 'item_id']].values]
        self.gallery = [tuple(x) for x in self.data[['fname', 'item_id']].values]
        pass
    
    def get_paths_from_names(self, fnames):
        paths = []
        for fname in fnames:
            paths.append(self.data[self.data.fname == fname].iloc[0].path)
        return paths


# ========================================================================

class TikiEmbeddingDataset(Dataset):
    
    def __init__(self, npy_path=None, features=None):
        assert npy_path is not None or features is not None, "Data not provided!"
        self.data = np.load(npy_path) if npy_path is not None else features
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return {'image' : torch.from_numpy(self.data[idx, :])}

    
# ========================================================================

class AutomaticTestDataset(Dataset):
    
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return {'image' : img}
