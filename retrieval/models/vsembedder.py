
import torch
import torch.nn as nn
import torchvision


class VSEmbedder(nn.Module):
    
    def __init__(self, embed_dim, device, dropout=None):
        
        super(VSEmbedder, self).__init__()
        
        base = torchvision.models.resnet50(pretrained=True)
        cnn_output_dim = base.fc.in_features
        base.fc = nn.Identity()
        
        self.base = base
        self.embed_dim = embed_dim
        self.standardize = nn.LayerNorm(cnn_output_dim, elementwise_affine=False)
        
        self.remap = None
        if embed_dim < cnn_output_dim:
            self.remap = nn.Linear(cnn_output_dim, embed_dim)
        
        self.device = device
        self = self.to(device)
        
    
    def forward(self, x):
        x = x.to(self.device)
        return self.get_embedding(x)
    
    
    def get_embedding(self, x):
        x = x.to(self.device)
        x_emb = self.base(x).view(x.shape[0], -1)
        x_emb = self.standardize(x_emb)
        if self.remap:
            x_emb = self.remap(x_emb)
        x_emb = nn.functional.normalize(x_emb, dim=1)
        return x_emb

    def freeze_base(self):
        for param in self.base.parameters():
            param.requires_grad = False
        pass
    
    def unfreeze_base(self):
        for param in self.base.parameters():
            param.requires_grad = True
        pass

    def load(self, checkpoint):
        self.load_state_dict(checkpoint)
        self = self.to(self.device)
        pass