import torch
from models.resnet import resnet50
from models.iresnet import iresnet100

from torch.nn import Parameter
import torch.nn as nn
import torch.nn.functional as F


class UncertaintyHead(nn.Module):
    ''' Evaluate the log(sigma^2) '''

    def __init__(self, in_feat=512):
        super(UncertaintyHead, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = torch.nn.Linear(in_feat,in_feat,bias=False)
        self.bn1 = nn.BatchNorm1d(in_feat, affine=True)
        self.fc2 = torch.nn.Linear(in_feat, in_feat,bias=False)
        self.bn2 = nn.BatchNorm1d(in_feat, affine=True)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.bn2(self.fc2(x)) # 2*log(sigma)
        x = torch.log(1e-6 + torch.exp(x))  # log(sigma^2)
        return x


class Baseline(torch.nn.Module):

    def __init__(self,
                 backbone="resnet50",
                 ):

        super(Baseline, self).__init__()
        assert backbone in ['resnet50','resnet101']
        self.backbone=backbone

        if self.backbone=="resnet50":
            self.encoder=resnet50()
            self.encoder.load_state_dict(torch.load("models/resnet50.pth"),strict=False)
            self.backbone_feature=2048

        elif self.backbone=='resnet101':
            self.encoder = iresnet100()
            self.encoder.load_state_dict(torch.load("models/resnet101.pth"),strict=False)
            self.backbone_feature = 512

        else :
            raise NotImplementedError("backbone must in ['resnet50', 'resnet101']")

    def forward(self, img):
        emb = self.encoder(img)
        return F.normalize(emb)