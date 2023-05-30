import os.path
import torch.nn
from datasets import ValidationDataset
import argparse
from torch.utils.data import DataLoader
from utils import *
from baseline_model import Baseline,UncertaintyHead
import torch.nn as nn
from losses import get_conf,ECELoss
from matplotlib import pyplot as plt
import math
import tqdm

FAR = 0.2


@torch.no_grad()
def getThresholdByFAR(preds, labels, FAR):

    neg_cnt = torch.sum((labels == 0.0).float())
    pos_cnt = labels.size(0) - neg_cnt
    ground_truth_label = labels
    predict_label = preds

    pos_dist = predict_label[ground_truth_label == 1.0]
    neg_dist = predict_label[ground_truth_label == 0.0]

    Ts = torch.cat([neg_dist, pos_dist], dim=0).cpu().numpy().tolist()

    thresholds = []
    for T in Ts:
        far = torch.sum((neg_dist > T).float()) / neg_cnt
        if far.item() <= FAR:
            thresholds.append(T)

    threshold = min(thresholds)
    return threshold


def getTAR(preds, labels, threshold):

    neg_cnt = torch.sum((labels == 0.0).float())
    pos_cnt = labels.size(0) - neg_cnt

    pos_dist = preds[labels == 1.0]
    tar = torch.sum((pos_dist > threshold).float()) / pos_cnt
    return tar.item()


def get_calibration_cos_simliarity(preds,
                                   calibration_param_w,
                                   calibration_param_b):
    theta = torch.arccos(preds)
    theta = torch.clamp(theta * calibration_param_w + calibration_param_b, min=0.0, max=np.pi)
    return torch.cos(theta)


def get_conf_uncertainty(x, y, sig2x, sig2y,
                         calibration_param_w,
                         calibration_param_b,
                         threshold, eps=1e-4):

    sig2s = torch.sum(x ** 2 * sig2y, dim=1, keepdim=False) + torch.sum(y ** 2 * sig2x, dim=1, keepdim=False)
    s = torch.cosine_similarity(x, y, dim=1)
    sig2sc = ((calibration_param_w ** 2) * (1 - get_calibration_cos_simliarity(s,calibration_param_w,calibration_param_b) ** 2+eps) / (1 - s ** 2 + eps)) * sig2s
    sig2c = 0.25 / ((1 - get_calibration_cos_simliarity(threshold,calibration_param_w,calibration_param_b)) ** 2 + eps) * sig2sc
    sig2c[s < threshold] = (0.25 / ((1 + get_calibration_cos_simliarity(threshold,calibration_param_w,calibration_param_b)) ** 2 + eps) * sig2sc)[s < threshold]
    sigc = torch.sqrt(sig2c)
    return sigc


def get_conf_with_uncertainty(conf, sigma, mean_sigma,factor=1):
    conf = conf - factor*(sigma-mean_sigma)
    return torch.clamp(conf, min=0.5, max=1.0)


def discard_low_conf(preds, labels, confs, rate=0):
    n, idxs = torch.topk(confs, largest=True, k=int(confs.size(0) * (1 - rate)))
    return preds[idxs], labels[idxs]


def evaluation(args):
    batch_size = args.batch_size
    model_root=args.model_root
    backbone=args.backbone
    image_size=args.image_size

    model = Baseline(backbone=backbone).cuda()
    model.encoder.load_state_dict(torch.load(os.path.join(model_root, "best.pkl")))
    model.eval()

    uncer_head = UncertaintyHead(
        in_feat=model.backbone_feature).cuda()
    uncer_head.load_state_dict(torch.load(os.path.join(model_root,args.head_checkpoint_path)))
    uncer_head.eval()

    wb = torch.load(os.path.join(model_root, "wb.pkl"))
    calibration_param_w, calibration_param_b=wb['weights'],wb['bias']

    logger = get_logger(os.path.join(model_root,"eval_uncertainty.log"))

    recalibration_dataset = ValidationDataset(sample_path="data/FIW/pairs/val.txt"
                                              ,images_size=image_size)

    test_dataset = ValidationDataset(sample_path="data/FIW/pairs/test.txt"
                                     ,images_size=image_size)

    recalibration_loader = DataLoader(recalibration_dataset, batch_size=batch_size, num_workers=4, pin_memory=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4, pin_memory=False)

    recalibration_mu1=[]
    recalibration_mu2=[]

    recalibration_sigma1=[]
    recalibration_sigma2=[]

    recalibration_labels=[]

    with torch.no_grad():
        for img1, img2, kin_class, label in tqdm.tqdm(recalibration_loader):
            emb1 = model(img1.cuda())
            emb2 = model(img2.cuda())

            sigma1=torch.exp(uncer_head(emb1))
            sigma2=torch.exp(uncer_head(emb2))

            recalibration_mu1.append(emb1.cpu())
            recalibration_mu2.append(emb2.cpu())

            recalibration_sigma1.append(sigma1.cpu())
            recalibration_sigma2.append(sigma2.cpu())

            recalibration_labels.append(label.cpu())

    torch.cuda.empty_cache()

    recalibration_mu1 = torch.cat(recalibration_mu1, dim=0).cuda()
    recalibration_mu2 = torch.cat(recalibration_mu2, dim=0).cuda()

    recalibration_sigma1 = torch.cat(recalibration_sigma1, dim=0).cuda()
    recalibration_sigma2 = torch.cat(recalibration_sigma2, dim=0).cuda()

    recalibration_labels = torch.cat(recalibration_labels,dim=0).cuda()
    recalibration_preds = torch.cosine_similarity(recalibration_mu1,recalibration_mu2,dim=1)

    before_calibration_threshold = getThresholdByFAR(recalibration_preds,recalibration_labels,FAR)
    recalibration_mean_uncertainty=torch.mean(get_conf_uncertainty(recalibration_mu1,
                                                        recalibration_mu2,
                                                        recalibration_sigma1,
                                                        recalibration_sigma2,
                                                        calibration_param_w,
                                                        calibration_param_b,
                                                        torch.from_numpy(np.array(before_calibration_threshold)).cuda()
                                                        ))

    logger.info("threshold before calibration: %.6f" % before_calibration_threshold)
    logger.info("recalibration_mean_uncertainty: %.6f" % recalibration_mean_uncertainty.item())

    test_mu1 = []
    test_mu2 = []

    test_sigma1 = []
    test_sigma2 = []

    test_labels = []

    with torch.no_grad():
        for img1, img2, kin_class, label in tqdm.tqdm(test_loader):
            emb1 = model(img1.cuda())
            emb2 = model(img2.cuda())

            sigma1 = torch.exp(uncer_head(emb1))
            sigma2 = torch.exp(uncer_head(emb2))

            test_mu1.append(emb1.cpu())
            test_mu2.append(emb2.cpu())

            test_sigma1.append(sigma1.cpu())
            test_sigma2.append(sigma2.cpu())

            test_labels.append(label.cpu())

    torch.cuda.empty_cache()

    test_mu1 = torch.cat(test_mu1, dim=0).cuda()
    test_mu2 = torch.cat(test_mu2, dim=0).cuda()

    test_sigma1 = torch.cat(test_sigma1, dim=0).cuda()
    test_sigma2 = torch.cat(test_sigma2, dim=0).cuda()

    test_labels = torch.cat(test_labels, dim=0).cuda()
    test_preds = torch.cosine_similarity(test_mu1, test_mu2, dim=1)

    test_confs_uncertainty=get_conf_uncertainty(test_mu1,
                                               test_mu2,
                                               test_sigma1,
                                               test_sigma2,
                                               calibration_param_w,
                                               calibration_param_b,
                                               torch.from_numpy(np.array(before_calibration_threshold)).cuda())

    test_calibration_preds=get_calibration_cos_simliarity(test_preds,calibration_param_w,calibration_param_b)

    test_calibration_threshold=get_calibration_cos_simliarity(torch.from_numpy(np.array(before_calibration_threshold)).cuda(),
                                                          calibration_param_w,calibration_param_b).item()

    test_uncalibration_confs=get_conf(test_preds,before_calibration_threshold)

    test_calibration_confs=get_conf(test_calibration_preds,test_calibration_threshold)
    test_acc = ((test_calibration_preds >= test_calibration_threshold).float() == test_labels).float()

    alpha1_confs=get_conf_with_uncertainty(test_calibration_confs,test_confs_uncertainty,recalibration_mean_uncertainty,factor=1)
    alpha2_confs=get_conf_with_uncertainty(test_calibration_confs,test_confs_uncertainty,recalibration_mean_uncertainty,factor=2)
    alpha3_confs=get_conf_with_uncertainty(test_calibration_confs,test_confs_uncertainty,recalibration_mean_uncertainty,factor=3)

    logger.info("uncalibration ECE: %.6f" % ECELoss()(test_uncalibration_confs,test_acc))
    logger.info("calibration ECE: %.6f" % ECELoss()(test_calibration_confs,test_acc))
    logger.info("alpha1 ECE: %.6f" % ECELoss()(alpha1_confs,test_acc))
    logger.info("alpha2 ECE: %.6f" % ECELoss()(alpha2_confs,test_acc))
    logger.info("alpha3 ECE: %.6f" % ECELoss()(alpha3_confs,test_acc))

    draw_uncertainty_vs_confidence_plot(test_calibration_confs,test_confs_uncertainty,os.path.join(model_root,'uncertainty_vx_confs.png'))

    draw_error_vs_reject_plot(test_calibration_preds,
                              test_calibration_confs,
                              alpha1_confs,
                              alpha2_confs,
                              alpha3_confs,
                              test_labels,
                              os.path.join(model_root, 'error_vx_reject.png')
                              )



def draw_uncertainty_vs_confidence_plot(calibration_confs,uncertainty,path):
    points = []
    n_bins = 10
    bin_boundaries = torch.linspace(0.5, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):

        in_bin = calibration_confs.gt(bin_lower.item()) * calibration_confs.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean()

        if prop_in_bin.item() > 0:
            # accuracy_in_bin = acc[in_bin].float().mean()
            avg_confidence_in_bin = calibration_confs[in_bin].mean()
            avg_sigs_in_bin = uncertainty[in_bin].mean()
            error_bar = uncertainty[in_bin].var()
            points.append([avg_confidence_in_bin.item(), avg_sigs_in_bin.item(), math.sqrt(error_bar.item())])

    x = [point[0] for point in points]
    y = [point[1] for point in points]
    err = [point[2] for point in points]

    plt.xlim(0.5, 1.0)
    plt.xlabel("Calibration confidence")
    plt.ylabel("Uncertainty")
    plt.errorbar(x, y, err, fmt='o:', ecolor='hotpink',
                 elinewidth=3, ms=5, mfc='wheat', mec='salmon', capsize=3)
    plt.savefig(path)
    plt.clf()


def draw_error_vs_reject_plot(cal_sim,
                              cal_confs,
                              alpha1_confs,
                              alpha2_confs,
                              alpha3_confs,
                              labels,
                              path
                              ):

    rs = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]

    points_cal = []
    for r in rs:
        sub_preds, sub_labels = discard_low_conf(cal_sim, labels, cal_confs, r)
        points_cal.append(1 - getTAR(sub_preds, sub_labels, getThresholdByFAR(sub_preds, sub_labels, FAR)))

    points_alpha1 = []
    for r in rs:
        sub_preds, sub_labels = discard_low_conf(cal_sim, labels, alpha1_confs, r)
        points_alpha1.append(1 - getTAR(sub_preds, sub_labels, getThresholdByFAR(sub_preds, sub_labels, FAR)))

    points_alpha2 = []
    for r in rs:
        sub_preds, sub_labels = discard_low_conf(cal_sim, labels, alpha2_confs, r)
        points_alpha2.append(1 - getTAR(sub_preds, sub_labels, getThresholdByFAR(sub_preds, sub_labels, FAR)))

    points_alpha3 = []
    for r in rs:
        sub_preds, sub_labels = discard_low_conf(cal_sim, labels, alpha3_confs, r)
        points_alpha3.append(1 - getTAR(sub_preds, sub_labels, getThresholdByFAR(sub_preds, sub_labels, FAR)))

    plt.xlim(0., 1.0)
    plt.xlabel("Filter Our Rate")
    plt.ylabel("FNMR@FMR={}".format(FAR))
    plt.plot(rs, points_cal, label=r'$\tilde{\mathcal{C}}_{w}^{0}$')
    plt.plot(rs, points_alpha1, label=r'$\tilde{\mathcal{C}}_{w}^{1}$')
    plt.plot(rs, points_alpha2, label=r'$\tilde{\mathcal{C}}_{w}^{2}$')
    plt.plot(rs, points_alpha3, label=r'$\tilde{\mathcal{C}}_{w}^{3}$')
    plt.legend()
    plt.savefig(path)
    plt.clf()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train_uncertainty")
    parser.add_argument("--batch_size", type=int, default=60, help="batch size")
    parser.add_argument("--model_root", type=str,default="infonce_resnet101")
    parser.add_argument("--image_size", type=int, default=112)
    parser.add_argument("--backbone", type=str, choices=['resnet50', 'resnet101'], default="resnet101")
    parser.add_argument("--head_checkpoint_path", type=str, default="heads/epoch_60_head.pkl")
    parser.add_argument("--gpu", default="0", type=str, help="gpu id you use")

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    torch.multiprocessing.set_start_method("spawn")
    set_seed(seed=100)
    evaluation(args)