import torch
import numpy as np
import random


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False


def save_config(config, file_path):
    fo = open(file_path, "w")
    fo.write("Configurtaions:\n")
    fo.write(str(config))
    fo.close()


def save_best_record(file_path, epoch, val_acc_lst, test_acc_lst):
    fo = open(file_path, "w")
    fo.write("Step: {}\n".format(epoch))

    topk = [1, 3, 5]
    for i in range(3):
        fo.write("Val top-{}: {:.4f}\n".format(topk[i], val_acc_lst[i]))
    for i in range(3):
        fo.write("Test top-{}: {:.4f}\n".format(topk[i], test_acc_lst[i]))

    fo.close()


def topk_correct(output, target, topk=(1,)):
    maxk = max(topk)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    return [torch.flatten(correct[:k]).float().sum(0).item() for k in topk]
