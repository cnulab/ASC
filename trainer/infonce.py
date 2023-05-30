import torch.nn
from sklearn.metrics import roc_curve
from datasets import InfoNCEDataset,ValidationDataset
from torch.optim import SGD,Adam
from losses import *
import argparse
from torch.utils.data import DataLoader
from utils import *
from baseline_model import Baseline


def training(args):

    batch_size = args.batch_size

    epochs = args.epochs
    train_steps = args.train_steps
    backbone = args.backbone
    image_size=args.image_size

    save_dir = "infonce_{}".format(backbone)

    os.makedirs(save_dir,exist_ok=True)

    log_path = os.path.join(save_dir,"train.log")
    logger = get_logger(log_path)


    model = Baseline(backbone=backbone).cuda()
    train_dataset = InfoNCEDataset( n=batch_size*train_steps,
                                    pairs_path="pairs/train.txt",
                                    image_size=image_size)

    val_dataset = ValidationDataset(sample_path="data/FIW/pairs/val_choose.txt",
                                images_size=image_size
                                )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=2, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=2, pin_memory=False)


    if args.optimizer=='sgd':
        optimizer_model = SGD(model.parameters(),lr=1e-4,momentum=0.9)
    else:
        optimizer_model = Adam(model.parameters(),lr=1e-5)


    criterion = InfoNCELoss(batch_size).cuda()


    max_acc=0.0

    for epoch_i in range(1,epochs+1):

        logger.info('epoch ' + str(epoch_i ))
        contrastive_loss_epoch = []

        model.train()
        for index_i, data in enumerate(train_loader):
            img1, img2,kin_class,label = data

            emb1 = model(img1.cuda())
            emb2 = model(img2.cuda())

            loss=criterion(emb1,emb2,args.temperature)

            optimizer_model.zero_grad()
            loss.backward()
            optimizer_model.step()

            contrastive_loss_epoch.append(loss.item())


        logger.info("infonce_loss:" + "%.6f" % np.mean(contrastive_loss_epoch))

        model.eval()
        acc,ece,threshold = val_model(model, val_loader)

        logger.info("acc is %.6f " % acc)
        logger.info("ece is %.6f " % ece)
        logger.info("threshold is %.6f " % threshold)

        if max_acc < acc:
            logger.info("validation acc improve from :" + "%.6f" % max_acc + " to %.6f" % acc)
            max_acc = acc
            save_model(model, os.path.join(save_dir, "best.pkl"))
        else:
            logger.info("validation acc did not improve from %.6f" % float(max_acc))


def save_model(model, path):
    torch.save(model.encoder.state_dict(), path)



@torch.no_grad()
def val_model(model, val_loader):
    y_true = []
    y_pred = []

    for img1, img2, kin_class,labels in val_loader:
        e1 = model(img1.cuda())
        e2 = model(img2.cuda())

        pred=torch.cosine_similarity(e1, e2, dim=1)

        y_pred.append(pred)
        y_true.append(labels)


    y_true = torch.cat(y_true, dim=0)
    y_pred = torch.cat(y_pred, dim=0)

    fpr, tpr, thresholds_keras = roc_curve(y_true.view(-1).cpu().numpy().tolist(),
                                           y_pred.view(-1).cpu().numpy().tolist())

    maxindex = (tpr - fpr).tolist().index(max(tpr - fpr))
    threshold = thresholds_keras[maxindex]

    acc = ((y_pred >= threshold).float() == y_true).float()
    conf = get_conf(y_pred, threshold)
    ece = ECELoss().cuda()(conf,acc).item()

    return torch.mean(acc).item(),ece,threshold



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train")

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--train_steps", type=int, default=50,help="number of iterations per epoch")
    parser.add_argument("--image_size", type=int, default=112, help="epochs number")

    parser.add_argument("--temperature", default=0.08, type=float,help="infoNCE temperature parameters")

    parser.add_argument("--backbone", type=str,choices=['resnet50','resnet101'], default="resnet101")
    parser.add_argument("--optimizer", type=str, choices=['sgd', 'adam'], default="sgd")
    parser.add_argument("--gpu", default="0", type=str, help="gpu id you use")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    torch.multiprocessing.set_start_method("spawn")
    set_seed(seed=100)
    training(args)
