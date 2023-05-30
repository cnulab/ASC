
import os
import numpy as np
import torch
import argparse
import tqdm
from utils import set_seed
from datasets import ValidationDataset,DataLoader
from sklearn.metrics import roc_curve
from losses import get_conf
from losses import ECELoss
from baseline_model import Baseline
from torch import nn
import torch.nn.functional as F
from utils import get_logger
from utils import np2tensor
import matplotlib.pyplot as plt


def draw_acc_vs_confs_plot(confs ,acc ,save_img,n_bins = 20 ):
    points = []
    bin_boundaries = torch.linspace(0.5, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):

        # Calculated |confidence - accuracy| in each bin
        in_bin = confs.gt(bin_lower.item()) * confs.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean()
        if prop_in_bin.item() > 0:
            accuracy_in_bin = acc[in_bin].float().mean()
            avg_confidence_in_bin = confs[in_bin].mean()
            points.append([avg_confidence_in_bin.item(), accuracy_in_bin.item()])

    x = [point[0] for point in points]
    y = [point[1] for point in points]

    plt.axis([0.5, 1.0, 0.5, 1.0])
    plt.xlabel("confidence")
    plt.ylabel("accuracy")
    plt.plot(x, y)
    plt.plot(list(np.arange(0.5, 1.05, 0.1)), list(np.arange(0.5, 1.05, 0.1)))
    plt.savefig(save_img)
    plt.clf()


def calibration(args):

    model_root=args.model_root
    backbone=args.backbone

    ori_model=Baseline(backbone=backbone).cuda()

    ori_model.encoder.load_state_dict(torch.load(os.path.join(model_root,'best.pkl')))

    args.logger = get_logger(os.path.join(model_root,"calibration.log"))

    model = ModelWithCalibration(ori_model)

    recalibration_dataset = ValidationDataset(sample_path="data/FIW/pairs/val.txt"
                                     ,images_size=args.image_size)

    test_dataset = ValidationDataset(sample_path="data/FIW/pairs/test.txt"
                                        ,images_size=args.image_size)

    recalibration_loader = DataLoader(recalibration_dataset, batch_size=args.batch_size, num_workers=2, pin_memory=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=2, pin_memory=False)

    model.angular_scaling(args,recalibration_loader)
    model.save_params(os.path.join(model_root,'wb.pkl'))

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


    def angular_scaling(self,args, recalibration_loader):
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
            for img1, img2, kin_class, label in tqdm.tqdm(recalibration_loader):
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

        self.best_ece = self.ece_criterion(confs_before_calibration,acc_before_calibration)

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

        args.logger.info("ECE before calibration: %.6f" % ece_before.item())

        draw_acc_vs_confs_plot(confs_before, acc_before, os.path.join(args.model_root, "before_calibration.jpg"))


        preds_after=self.get_calibration_cos_simliarity(preds)

        threshold_after=self.get_calibration_cos_simliarity(np2tensor(np.array(self.threshold_before_calibration))).item()

        args.logger.info("threshold after calibration: %.6f" % threshold_after)

        confs_after=self.get_conf(preds_after,threshold_after)

        acc_after = ((preds_after>= threshold_after).float() == labels).float()

        args.logger.info("accuracy after calibration: %.6f"% torch.mean(acc_after).item())

        ece_after=self.ece_criterion(confs_after,acc_after)

        args.logger.info("ECE after calibration: %.6f"% ece_after.item())

        draw_acc_vs_confs_plot(confs_after,acc_after,os.path.join(args.model_root,"after_calibration.jpg"))


    def save_params(self,path='wb.pkl'):
        torch.save({"weights":self.weights,"bias":self.bias},path)

    def load_params(self,path='wb.pkl'):
        wb=torch.load(path)
        self.weights.data.copy_(wb['weights'])
        self.bias.data.copy_(wb['bias'])
      

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="train_encoder")
    parser.add_argument("--model_root", type=str, required=True,
                        help='path of the model you want to load')

    parser.add_argument("--image_size", type=int, default=112)
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
