
import argparse
import os
import torch
from utils import set_seed



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train")

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--image_size", type=int, default=112)
    parser.add_argument("--train_steps", type=int, default=50,help="number of iterations per epoch")

    parser.add_argument("--loss", type=str,choices=['infonce','arcface','softmax','triplet'], default="triplet")

    parser.add_argument("--backbone", type=str,choices=['resnet50','resnet101'], default="resnet101")
    parser.add_argument("--optimizer", type=str, choices=['sgd', 'adam'], default="adam")
    parser.add_argument("--gpu", default="0", type=str, help="gpu id you use")

    #infonce
    parser.add_argument("--temperature", default=0.08, type=float,help="infoNCE temperature parameters")

    #triplet
    parser.add_argument("--margin", default=0.25, type=float, help="Triplet distance margiin")
    parser.add_argument("--use_semihard_negatives", type=bool, default=False)

    #arcface
    parser.add_argument("--s", default=10.0, type=float)
    parser.add_argument("--m", default=0.40, type=float)

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    torch.multiprocessing.set_start_method("spawn")
    set_seed(seed=100)

    trainer=__import__("trainer",fromlist=[args.loss])
    exec("trainer.{}.training(args)".format(args.loss))