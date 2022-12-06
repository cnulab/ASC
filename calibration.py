

import os
import numpy as np
import torch

import argparse

import tqdm

from utils import set_seed
from datasets import ValAndTest,DataLoader

from sklearn.metrics import roc_curve
from losses import get_conf
from losses import ECELoss
from files import Dir
from backbones import Backbone
from torch import nn
import torch.nn.functional as F
from draw_pic import draw_distance,draw_number
from utils import get_logger
from utils import np2tensor



def calibration(args):


    ori_model=Backbone(backbone=args.backbone).cuda()
    ori_model.encoder.load_state_dict(torch.load(args.model_path))

    args.save_dir=Dir.mkdir(args.save_dir)
    args.logger = get_logger(os.path.join(args.save_dir,"calibration.log"))

    model = ModelWithCalibration(ori_model)

    calibration_dataset = ValAndTest(sample_path="data/FIW/pairs/val.txt"
                                     ,images_size=ori_model.imagesize,backbone=args.backbone)
    test_dataset = ValAndTest(sample_path="data/FIW/pairs/test.txt"
                              ,images_size=ori_model.imagesize,backbone=args.backbone)

    calibration_loader = DataLoader(calibration_dataset, batch_size=args.batch_size, num_workers=2, pin_memory=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=2, pin_memory=False)

    model.angular_scaling(args,calibration_loader)
    model.test(args,test_loader)



class ModelWithCalibration(nn.Module):

    def __init__(self, model):
        super(ModelWithCalibration, self).__init__()

        self.model=model
        self.model.eval()

        self.weights = nn.Parameter(torch.ones((1,)))
        self.bias=nn.Parameter(torch.zeros((1,)))

        self.best_weights=torch.clone(self.weights.data)
        self.best_bias = torch.clone(self.bias.data)

        self.get_conf=get_conf
        self.ece_criterion = ECELoss().cuda()
        self.mse_criterion = nn.MSELoss().cuda()


    def forward(self, img1,img2):
        emb1 = self.model(img1)
        emb2 = self.model(img2)
        emb1 = F.normalize(emb1)
        emb2 = F.normalize(emb2)
        return self.get_calibration_cos_simliarity(torch.cosine_similarity(emb1,emb2,dim=1))



    def get_calibration_cos_simliarity(self, cosine):
        theta = torch.arccos(cosine)
        theta = torch.clamp(theta*self.weights+self.bias, min=0.0, max=np.pi)
        return torch.cos(theta)


    def angular_scaling(self,args,calibration_loader):
        """
        Tune the tempearature of the model (using the validation set).
        We're going to set it to optimize NLL.
        valid_loader (DataLoader): validation set loader
        """

        self.cuda()

        labels = []
        preds = []

        print("load calibration features")
        with torch.no_grad():
            for img1, img2, kin_class, label in tqdm.tqdm(calibration_loader):
                e1 = self.model(img1.cuda())
                e2 = self.model(img2.cuda())

                pred = torch.cosine_similarity(e1, e2, dim=1)
                preds.append(pred)
                labels.append(label)

        preds = torch.cat(preds, dim=0)
        labels = torch.cat(labels, dim=0)


        labels=labels.view(-1)
        preds=preds.view(-1)

        fpr, tpr, thresholds = roc_curve(labels.cpu().numpy().tolist(),
                                               preds.cpu().numpy().tolist())

        maxindex = (tpr - fpr).tolist().index(max(tpr - fpr))
        self.threshold_before_calibration = thresholds[maxindex]


        confs_before_calibration=self.get_conf(preds,self.threshold_before_calibration)

        acc_before_calibration = ((preds >=self.threshold_before_calibration).float() == labels).float()

        self.best_ece=self.ece_criterion(confs_before_calibration,acc_before_calibration)

        optimizer = torch.optim.LBFGS([self.weights,self.bias], lr=args.lr, max_iter=args.max_iter)

        cos_labels=2*labels-1 #label {0,1} to {1,-1}

        def get_threshold(preds, labels):
            fpr, tpr, thresholds_keras = roc_curve(labels.cpu().numpy().tolist(),
                                                   preds.cpu().numpy().tolist())

            maxindex = (tpr - fpr).tolist().index(max(tpr - fpr))
            threshold = thresholds_keras[maxindex]
            return threshold


        def eval_scale():
            optimizer.zero_grad()
            new_preds=self.get_calibration_cos_simliarity(preds)
            loss = self.mse_criterion(new_preds,cos_labels)
            loss.backward()

            with torch.no_grad():
                threshold = get_threshold(new_preds, labels)
                new_conf = self.get_conf(new_preds, threshold)
                new_acc = ((new_preds >= threshold).float() == labels).float()
                now_ece = self.ece_criterion(new_conf, new_acc)

                if self.best_ece > now_ece:
                    self.best_ece = now_ece
                    self.best_bias = torch.clone(self.bias.data)
                    self.best_weights = torch.clone(self.weights.data)
            return loss

        optimizer.step(eval_scale)
        self.weights.data.copy_(self.best_weights)
        self.bias.data.copy_(self.best_bias)


    
    @torch.no_grad()
    def test(self,args,test_loader):
        """
        Tune the tempearature of the model (using the validation set).
        We're going to set it to optimize NLL.
        valid_loader (DataLoader): validation set loader
        """
        self.cuda()

        labels = []
        preds = []

        print("load test features")
        with torch.no_grad():
            for img1, img2, kin_class, label in tqdm.tqdm(test_loader):
                e1 = self.model(img1.cuda())
                e2 = self.model(img2.cuda())

                pred = torch.cosine_similarity(e1, e2, dim=1)
                preds.append(pred)
                labels.append(label)

        preds = torch.cat(preds, dim=0)
        labels = torch.cat(labels, dim=0)


        args.logger.info("threshold before calibration: %.6f" % self.threshold_before_calibration)

        confs_before = self.get_conf(preds, self.threshold_before_calibration)

        acc_before= ((preds >= self.threshold_before_calibration).float() ==labels).float()

        args.logger.info("accuracy before calibration: %.6f" % torch.mean(acc_before).item())

        ece_before = self.ece_criterion(confs_before, acc_before)

        args.logger.info("ECE before calibration: %.6f"% ece_before.item())

        draw_distance(confs_before, acc_before, os.path.join(args.save_dir, "before_calibration.jpg"))


        preds_after=self.get_calibration_cos_simliarity(preds)

        threshold_after=self.get_calibration_cos_simliarity(np2tensor(np.array(self.threshold_before_calibration))).item()

        args.logger.info("threshold after calibration: %.6f" % threshold_after)

        confs_after=self.get_conf(preds_after,threshold_after)

        acc_after = ((preds_after>= threshold_after).float() == labels).float()

        args.logger.info("accuracy after calibration: %.6f"% torch.mean(acc_after).item())

        ece_after=self.ece_criterion(confs_after,acc_after)

        args.logger.info("ECE after calibration: %.6f"% ece_after.item())

        draw_distance(confs_after,acc_after,os.path.join(args.save_dir,"after_calibration.jpg"))
    
    def save_params(self,path='wb.pkl'):
        torch.save({"weights":self.weights,"bias":self.bias},path)
    
    def load_params(self,path='wb.pkl'):
        wb=torch.load(path)
        self.weights.data.copy_(wb['weights'])
        self.bias.data.copy_(wb['bias'])
      

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="train_encoder")
    parser.add_argument("--model_path", type=str, required=True,
                        help='path of the model you want to load')

    parser.add_argument("--save_dir", type=str, default="calibration", help='dir where you want to save the results')
    parser.add_argument("--backbone", type=str,choices=['resnet50','resnet101'], default="resnet101")

    parser.add_argument("--lr", default=0.1, type=float,help="learning rate")
    parser.add_argument("--max_iter", default=1000, type=int, help="maximum number of iterations")
    parser.add_argument("--batch_size", type=int, default=60, help="batch size")
    parser.add_argument("--gpu", default="0", type=str, help="gpu id you use")

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    torch.multiprocessing.set_start_method("spawn")
    set_seed(seed=100)
    calibration(args)
