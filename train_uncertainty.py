import os.path
import torch.nn
from datasets import TrainUncertaintyDataset
import argparse
from torch.utils.data import DataLoader
from losses import MLSLoss
from utils import *
from baseline_model import Baseline, UncertaintyHead
import os


def training(args):
    batch_size = args.batch_size

    train_steps = args.train_steps

    model_root = args.model_root
    backbone = args.backbone

    epochs = args.epochs

    checkpoints_root = os.path.join(model_root, args.checkpoints_root)
    os.makedirs(checkpoints_root, exist_ok=True)

    log_path = os.path.join(model_root, "train_uncertainty.log")
    logger = get_logger(log_path)

    train_dataset = TrainUncertaintyDataset(n=batch_size * train_steps, sample_path="data/FIW/pairs/train.txt",
                                            images_size=args.image_size)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=2, pin_memory=False)

    model = Baseline(backbone=backbone).cuda()
    model.encoder.load_state_dict(torch.load(os.path.join(model_root, "best.pkl")))
    model.eval()

    uncer_head = UncertaintyHead(
        in_feat=model.backbone_feature).cuda()

    optimizer = torch.optim.SGD([{'params': uncer_head.parameters()}], lr=1e-3, momentum=0.9)

    criterion = MLSLoss().cuda()

    for epoch_i in range(1, epochs + 1):

        logger.info('epoch ' + str(epoch_i))
        loss_epoch = []
        uncer_head.train()

        for index_i, data in enumerate(train_loader):
            img1, img2 = data

            with torch.no_grad():
                em1 = model(img1.cuda())
                em2 = model(img2.cuda())

            log_sig1 = uncer_head(em1)
            log_sig2 = uncer_head(em2)

            loss = criterion(em1, em2, log_sig1, log_sig2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_epoch.append(loss.item())

        logger.info("loss:" + "%.6f" % np.mean(loss_epoch))

        if epoch_i % 5 == 0:
            save_model(uncer_head, os.path.join(checkpoints_root, "epoch_" + str(epoch_i) + "_head.pkl"))


def save_model(model, path):
    torch.save(model.state_dict(), path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train_uncertainty")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--epochs", type=int, default=60, help="epochs number default 80")
    parser.add_argument("--train_steps", type=int, default=100)

    parser.add_argument("--model_root", type=str, default="infonce_resnet101")
    parser.add_argument("--image_size", type=int, default=112)
    parser.add_argument("--backbone", type=str, choices=['resnet50', 'resnet101'], default="resnet101")
    parser.add_argument("--gpu", default="0", type=str, help="gpu id you use")

    parser.add_argument("--checkpoints_root", type=str, default="heads")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    torch.multiprocessing.set_start_method("spawn")
    set_seed(seed=100)
    training(args)