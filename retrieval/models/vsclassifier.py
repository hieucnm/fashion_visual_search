import torch
import torch.nn as nn
import torchvision


class VSClassifier(nn.Module):
    
    def __init__(self, output_dim, device, dropout=None, uncertainty=False):
        
        super(VSClassifier, self).__init__()
               
        base = torchvision.models.resnet50(pretrained=True)
        cnn_output_dim = base.fc.in_features
        base.fc = nn.Identity()
        
        self.output_dim = output_dim + 1 if uncertainty else output_dim
        self.base = base
        
        if dropout:
            self.fc = nn.Sequential(
                nn.Dropout(float(dropout)),
                nn.Linear(cnn_output_dim, self.output_dim)
            )
        else:
            self.fc = nn.Linear(cnn_output_dim, self.output_dim)

        self.device = device
        self = self.to(device)
        
        
    # thử init He: với w là layer.weight
    # nn.init.kaiming_normal_(w, mode='fan_out', nonlinearity='relu')
    
    def forward(self, x):
        x = x.to(self.device)
        out = self.base(x)
        out = self.fc(out)
        return out
    
    def freeze_base(self):
        for param in self.base.parameters():
            param.requires_grad = False
        return
    
    def unfreeze_base(self):
        for param in self.base.parameters():
            param.requires_grad = True
        return
