import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import warnings
import copy
from utils import save_net
from torch.autograd import Variable
from pytorch_msssim import ssim
warnings.filterwarnings("ignore")


def train(args, model, vgg_model, train_dataloader, val_dataloader):
    device = args.device

    params = model.parameters()

    optimizer = optim.Adam(params, lr=args.lr, betas=[args.beta1, args.beta2],
                           weight_decay=args.weight_decay)

    # mae = nn.L1Loss()
    mse = nn.MSELoss()
    best_loss = np.inf

    for epoch in tqdm(range(args.epochs)):
        print('-' * 10)
        print(f'Epoch {epoch + 1}/{args.epochs}')

        batch_train_loss = []

        model.train()
        pbar = tqdm(train_dataloader, total=len(train_dataloader))
        for img, label in pbar:
            train_x = img.to(device)
            label = label.to(device)

            pred = model(train_x)

            vgg_model.eval()
            with torch.no_grad():
                pred2 = vgg_model(pred)
                label2 = vgg_model(label)

            pred_var = Variable(pred)
            label_var = Variable(label)
            loss1 = 1 - ssim(label_var, pred_var)

            loss2 = mse(label2, pred2)

            loss3 = mse(label, pred)

            loss = loss1 + loss3 + 0.1*loss2
            batch_train_loss.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix({'Epoch': epoch+1,
                              'Train Loss': loss.item()})

            torch.cuda.empty_cache()

        batch_val_loss = []

        model.eval()
        with torch.no_grad():
            pbar = tqdm(val_dataloader, total=len(val_dataloader))
            for img, label in pbar:
                val_x = img.to(device)
                val_label = label.to(device)

                val_pred = model(val_x)

                vgg_model.eval()
                with torch.no_grad():
                    val_pred2 = vgg_model(val_pred)
                    val_label2 = vgg_model(val_label)

                val_pred_var = Variable(val_pred)
                val_label_var = Variable(val_label)
                loss1 = 1 - ssim(val_label_var, val_pred_var)

                loss2 = mse(val_label2, val_pred2)

                loss3 = mse(val_label, val_pred)

                loss_val = loss1 + loss3 + 0.1*loss2

                batch_val_loss.append(loss_val.item())

                pbar.set_postfix({'Epoch': epoch + 1,
                                  'Val Loss': loss_val.item()})

                del val_x, val_label, val_pred
                torch.cuda.empty_cache()

        mean_train_loss = np.mean(batch_train_loss)
        mean_val_loss = np.mean(batch_val_loss)

        print(f'\nEpoch train loss: {mean_train_loss}, Epoch val loss: {mean_val_loss}')

        if mean_val_loss < best_loss:
            best_loss = mean_val_loss
            best_weights = copy.deepcopy(model.state_dict())
            best_weights_path = save_net(args.saveM_path, best_weights, str(epoch+1))

    print(f'\nBest validation loss: {best_loss}')
    print(f'\nBest weights can be found here: {best_weights_path}')
