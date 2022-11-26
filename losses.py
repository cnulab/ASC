import torch
from  torch import nn
import torch.nn.functional as F
import numpy as np
from torch.nn.modules.distance import PairwiseDistance
from torch.nn import Parameter
import math



def get_conf(s, threshold):
    conf = (threshold - s) / (threshold + 1)
    conf[s >= threshold] = ((s - threshold) / (1 - threshold))[s >= threshold]
    return conf*0.5+0.5



class ContrastiveLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x1, x2, tau=0.1):
        x1x2 = torch.cat([x1, x2], dim=0)
        x2x1 = torch.cat([x2, x1], dim=0)

        cosine_mat = torch.cosine_similarity(torch.unsqueeze(x1x2, dim=1),
                                             torch.unsqueeze(x1x2, dim=0), dim=2) / tau

        mask = torch.eye(x1.size(0))
        mask = torch.cat([mask, mask], dim=0)
        mask = 1.0 - torch.cat([mask, mask], dim=1)

        numerators = torch.exp(torch.cosine_similarity(x1x2, x2x1, dim=1) / tau)
        denominators = torch.sum(torch.exp(cosine_mat) * mask, dim=1)

        return -torch.mean(torch.log(numerators / denominators), dim=0)




class TripletLoss(nn.Module):

    def __init__(self, margin=0.0):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.pdist = PairwiseDistance(p=2)

    def forward(self, anchor, positive, negative):
        pos_dist = self.pdist.forward(anchor, positive)
        neg_dist = self.pdist.forward(anchor, negative)

        hinge_dist = torch.clamp(self.margin + pos_dist - neg_dist, min=0.0)
        loss = torch.mean(hinge_dist)
        return loss




class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin

            cos(theta + m)
        """

    def __init__(self, in_features, out_features, s=32.0, m=0.50):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s
        return output






class ECELoss(nn.Module):
    def __init__(self, n_bins=20):
        """
        n_bins (int): number of confidence interval bins
        """
        super(ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0.5, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, confs, accs):
        ece = torch.zeros(1, device=confs.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confs.gt(bin_lower.item()) * confs.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accs[in_bin].float().mean()
                avg_confidence_in_bin = confs[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        return ece

