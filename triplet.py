import torch.nn
from sklearn.metrics import roc_curve, auc
from datasets import TripletTrain,ValAndTest
from torch.optim import SGD,Adam
from losses import *
import argparse
from torch.utils.data import DataLoader
from files import Dir
from backbones import Backbone
from utils import *




def training(args):

    batch_size = args.batch_size
    val_batch_size = args.batch_size
    epochs = args.epochs

    train_steps = args.train_steps

    use_semihard_negatives=args.use_semihard_negatives
    margin = args.margin
    backbone=args.backbone

    save_dir = Dir.mkdir("triplet_" + backbone)
    log_path = os.path.join(save_dir, "train.log")
    logger = get_logger(log_path)

    model = Backbone(backbone=backbone).cuda()

    train_dataset = TripletTrain(
                 number_triplet=epochs*batch_size*train_steps,
                 triplet_batch_size=batch_size,
                backbone=backbone,
                images_size=model.imagesize
    )

    val_dataset = ValAndTest(sample_path='./data/FIW/pairs/val_choose.txt',
                             backbone=backbone,
                             images_size=model.imagesize
                             )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=1, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=val_batch_size, num_workers=1, pin_memory=False)


    if args.optimizer=='sgd':
        optimizer_model = SGD(model.parameters(),lr=1e-4,momentum=0.9)
    else:
        optimizer_model = Adam(model.parameters(),lr=1e-5)

    tri_citersion=TripletLoss(margin=margin).cuda()
    l2_distance = PairwiseDistance(p=2)
    max_acc=0.0

    for epoch_i in range(1,epochs+1):

        logger.info('epoch ' + str(epoch_i ))
        triplet_loss_epoch = []
        model.train()

        for index_i, data in enumerate(train_loader):
            anc_imgs , pos_imgs , neg_imgs = data

            anc_embeddings=model(anc_imgs)
            pos_embeddings=model(pos_imgs)
            neg_embeddings=model(neg_imgs)

            pos_dists = l2_distance.forward(anc_embeddings, pos_embeddings)
            neg_dists = l2_distance.forward(anc_embeddings, neg_embeddings)

            if use_semihard_negatives:
                # Semi-Hard Negative triplet selection
                #  (negative_distance - positive_distance < margin) AND (positive_distance < negative_distance)
                #   Based on: https://github.com/davidsandberg/facenet/blob/master/src/train_tripletloss.py#L295
                first_condition = (neg_dists - pos_dists < margin).cpu().numpy().flatten()
                second_condition = (pos_dists < neg_dists).cpu().numpy().flatten()
                all = (np.logical_and(first_condition, second_condition))
                valid_triplets = np.where(all == 1)
            else:
                # Hard Negative triplet selection
                #  (negative_distance - positive_distance < margin)
                #   Based on: https://github.com/davidsandberg/facenet/blob/master/src/train_tripletloss.py#L296
                all = (neg_dists - pos_dists < margin).cpu().numpy().flatten()
                valid_triplets = np.where(all == 1)

            anc_valid_embeddings = anc_embeddings[valid_triplets]
            pos_valid_embeddings = pos_embeddings[valid_triplets]
            neg_valid_embeddings = neg_embeddings[valid_triplets]

            triplet_loss = tri_citersion(
                anchor=anc_valid_embeddings,
                positive=pos_valid_embeddings,
                negative=neg_valid_embeddings
            )

            optimizer_model.zero_grad()
            triplet_loss.backward()
            optimizer_model.step()

            triplet_loss_epoch.append(triplet_loss.item())
            if index_i  == train_steps-1:
                break

        use_sample=epoch_i*batch_size*train_steps
        train_dataset.set_bias(use_sample)

        logger.info("triplet_loss:" + "%.6f" % np.mean(triplet_loss_epoch))

        model.eval()
        acc, ece, threshold = val_model(model, val_loader)

        logger.info("acc is %.6f " % acc)
        logger.info("ece is %.6f " % ece)
        logger.info("threshold is %.6f " % threshold)

        if max_acc < acc:
            logger.info("validation acc improve from :" + "%.6f" % max_acc + " to %.6f" % acc)
            max_acc = acc
            save_model(model, os.path.join(save_dir, "triplet_"+backbone+"_best_model.pkl"))
        else:
            logger.info("validation acc did not improve from %.6f" % float(max_acc))

    save_model(model, os.path.join(save_dir, "triplet_"+backbone+"_final_model.pkl"))



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
    parser.add_argument("--batch_size", type=int, default=45, help="batch size")
    parser.add_argument("--epochs", type=int, default=80, help="epochs number")
    parser.add_argument("--train_steps", type=int, default=50,help="number of iterations per epoch")

    parser.add_argument("--margin", default=0.25, type=float, help="Triplet distance margiin")
    parser.add_argument("--use_semihard_negatives", type=bool, default=False)

    parser.add_argument("--optimizer", type=str, choices=['sgd', 'adam'], default="adam")
    parser.add_argument("--backbone", type=str,choices=['resnet50','resnet101'], default="resnet101")
    parser.add_argument("--gpu", default="0", type=str, help="gpu id you use")

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    torch.multiprocessing.set_start_method("spawn")
    set_seed(seed=100)

    training(args)
