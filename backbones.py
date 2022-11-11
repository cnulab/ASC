import torch
from pretrain_models.torch_resnet50 import ResNet50
from pretrain_models.torch_resnet101 import ResNet101
import numpy as np
import torch.nn.functional as F



class Backbone(torch.nn.Module):

    def __init__(self,
                 backbone="resnet50",
                 pretrain=True,
                 ):

        super(Backbone, self).__init__()
        assert backbone in ['resnet50','resnet101']
        self.backbone=backbone

        if self.backbone=="resnet101":
            self.encoder= ResNet101("pretrain_models/resnet101.npy") if pretrain else ResNet101()
            self.backbone_feature=512
            self.imagesize=112
            print("load resnet101")

        elif self.backbone=='resnet50':
            self.encoder = ResNet50("pretrain_models/resnet50.npy") if pretrain else ResNet50()
            self.backbone_feature = 2048
            self.imagesize=224
            print("load resnet50")


    def forward(self, img):
        emb=self.encoder(img)
        return F.normalize(emb)


