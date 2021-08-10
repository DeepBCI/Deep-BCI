import os
from torch.utils.data import Dataset
import random
import argparse
import models
from train_eval import *
import utils

def Net(n_class, n_ch, n_time):
    embedding_net = models.EEGNet(n_class, n_ch, n_time)
    model = models.FcClfNet(embedding_net)
    return model

def exp(args, fold_idx, train_set, valid_set, test_set):
    path = args.save_root + args.result_dir
    if not os.path.isdir(path):
        os.makedirs(path)
        os.makedirs(path + '/models')

    with open(path + '/args.txt', 'w') as f:
        f.write(str(args))

    import torch.cuda
    cuda = torch.cuda.is_available()
    device = 'cuda' if cuda else 'cpu'

    if cuda:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    seed = args.seed
    random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, pin_memory=False)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=args.batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    temp = torch.utils.data.Subset(train_set, range(100, 200))

    model = Net(args.n_class, args.n_ch, args.n_time)

    mb_params = utils.param_size(model)
    print(f"Model size = {mb_params:.4f} MB")
    if cuda:
        model.cuda(device=device)
    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    results_columns = [f'valid_loss', f'test_loss', f'valid_accuracy', f'test_accuracy']
    df = pd.DataFrame(columns=results_columns)

    valid_min_loss = float('inf')
    best_acc_loss = 0

    for epochidx in range(1, args.epochs):
        print(epochidx)
        train(10, model, device, train_loader, optimizer, scheduler, cuda, args.gpuidx)
        valid_loss, valid_score, _ = eval(model, device, valid_loader)
        test_loss, test_score, _ = eval(model, device, test_loader)

        results = {f'valid_loss': valid_loss, f'test_loss': test_loss, f'valid_accuracy': valid_score,
                   f'test_accuracy': test_score}
        df = df.append(results, ignore_index=True)
        print(results)

        scheduler.step()
        lr = scheduler.get_last_lr()[0]

        print(f'LR : {lr}')
        if valid_loss < valid_min_loss:
            valid_min_loss = valid_loss
            best_acc_loss = test_score
            torch.save(model.state_dict(), os.path.join(
                path, 'models',
                f"model_fold{fold_idx}.pt"))
            best_loss_epoch = epochidx
        print(f'current best(loss) acc : {best_acc_loss:.4f} at epoch {best_loss_epoch}')

    best_model = Net(args.n_class, args.n_ch, args.n_time)
    best_model.load_state_dict(torch.load(os.path.join(
        path, 'models',
        f"model_fold{fold_idx}.pt"), map_location=device))
    if cuda:
        best_model.cuda(device=device)

    print("best accuracy")
    _, test_score, _ = eval(best_model, device, test_loader)

    utils.enablePrint()
    print(f"subject:{fold_idx}, acc:{test_score}")

    df = pd.DataFrame(np.array(test_score).reshape(-1, 1), columns=['sess2-on'])
    print(f"all acc: {np.mean(test_score):.4f}")

    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='openbmi_gigadb')
    parser.add_argument('--data-root', default='')
    parser.add_argument('--save-root', default='Result/')
    parser.add_argument('--result-dir', default='')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N')
    parser.add_argument('--test-batch-size', type=int, default=100, metavar='N')
    parser.add_argument('--epochs', type=int, default=100, metavar='N')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR')
    parser.add_argument('--save-model', action='store_true', default=True)
    parser.add_argument('--device', type=int, default=0,  metavar='N')

    args = parser.parse_args()
    args.gpuidx = 0
    import pandas as pd

    df_all = pd.DataFrame()
    args.seed = 2020
    target = np.r_[0:54]
    for fold_idx in target:
        train_set, valid_set, test_set, args = utils.get_data_subject_dependent_single_3cls_1s(args, fold_idx,opt=1)
        df = exp(args, fold_idx, train_set, valid_set, test_set)
        df_all = pd.concat([df_all, df], axis=0)
        print(df_all)
        print(df_all.mean())
