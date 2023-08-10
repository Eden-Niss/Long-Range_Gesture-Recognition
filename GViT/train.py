import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import warnings
import copy
from utils import save_net
from utils import get_idx
import time
import wandb

warnings.filterwarnings("ignore")


def train(args, model, train_dataloader, val_dataloader):
    device = args.device

    edge_index = get_idx(args.img_size, args.img_size).to(device)

    params = model.parameters()

    optimizer = optim.Adam(params, lr=args.lr, betas=[args.beta1, args.beta2],
                           weight_decay=args.weight_decay)

    criterion = nn.CrossEntropyLoss()
    best_acc = -np.inf

    for epoch in tqdm(range(args.epochs)):
        print('-' * 10)
        print(f'Epoch {epoch + 1}/{args.epochs}')

        batch_train_loss = []
        batch_train_acc = []

        model.train()
        pbar = tqdm(train_dataloader, total=len(train_dataloader))
        for img, label in pbar:
            train_x = img.to(device)
            label = label.to(device)

            pred = model(train_x, edge_index)

            loss = criterion(pred, label)
            batch_train_loss.append(loss.item())

            _, pred_class = torch.max(pred, dim=1)
            label_class = torch.nonzero(label == 1, as_tuple=False)[:, 1]
            acc = torch.sum(torch.eq(pred_class, label_class)).item() / len(label_class)
            batch_train_acc.append(acc)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix({'Epoch': epoch+1,
                              'Train Loss': loss.item(),
                              'Train Acc': acc})

            torch.cuda.empty_cache()

        batch_val_loss = []
        batch_val_acc = []

        model.eval()
        with torch.no_grad():
            pbar = tqdm(val_dataloader, total=len(val_dataloader))
            for img, label in pbar:
                val_x = img.to(device)
                val_label = label.to(device)

                val_pred = model(val_x, edge_index)

                loss_val = criterion(val_pred, val_label)
                batch_val_loss.append(loss_val.item())

                _, pred_class = torch.max(val_pred, dim=1)
                label_class = torch.nonzero(val_label == 1, as_tuple=False)[:, 1]
                acc_val = torch.sum(torch.eq(pred_class, label_class)).item() / len(label_class)
                batch_val_acc.append(acc_val)

                pbar.set_postfix({'Epoch': epoch + 1,
                                  'Val Loss': loss_val.item(),
                                  'Val Acc': acc_val})

                del val_x, val_label, val_pred
                torch.cuda.empty_cache()

        mean_train_loss = np.mean(batch_train_loss)
        mean_val_loss = np.mean(batch_val_loss)
        mean_train_acc = np.mean(batch_train_acc)
        mean_val_acc = np.mean(batch_val_acc)
        print(f'\nEpoch train loss: {mean_train_loss}, Epoch val loss: {mean_val_loss}')
        print(f'Epoch train acc: {mean_train_acc}, Epoch val acc: {mean_val_acc}')

        if mean_val_acc > best_acc:
            best_acc = mean_val_acc
            best_weights = copy.deepcopy(model.state_dict())
            best_weights_path = save_net(args.saveM_path, best_weights, str(epoch+1))

    print(f'\nBest validation accuracy: {best_acc}')
    print(f'\nBest weights can be found here: {best_weights_path}')
