import torch
import matplotlib.pyplot as plt
import numpy as np
import math



def draw_distance(confs ,acc ,save_img,n_bins = 20 ):
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




@torch.no_grad()
def draw_number( confs, acc, save_img, n_bins = 20):

    confs, index = torch.sort(confs, dim=0)
    acc = acc[index.view(-1)]
    sample = torch.cat([confs.view((-1, 1)), acc.view((-1, 1))], dim=1)

    sample = torch.split(sample, math.ceil(sample.size(0) / n_bins), dim=0)
    points = [torch.mean(sample[i], dim=0).cpu().numpy().tolist() for i in range(len(sample))]

    x = [point[0] for point in points]
    y = [point[1] for point in points]

    plt.axis([0.5, 1.0, 0.5, 1.0])
    plt.xlabel("confidence")
    plt.ylabel("accuracy")
    plt.plot(x, y)

    plt.plot(list(np.arange(0.5, 1.05, 0.1)), list(np.arange(0.5, 1.05, 0.1)))
    plt.savefig(save_img)
    plt.clf()
