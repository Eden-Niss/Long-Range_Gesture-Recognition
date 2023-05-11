import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import warnings
from torch.optim.lr_scheduler import ReduceLROnPlateau
import copy
from utils import save_net
import time
import wandb

warnings.filterwarnings("ignore")


class CNN(nn.Module):
    def __init__(self, device):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, 3),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 512, 3),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(512 * 5 * 5, 64),  # 12800
            nn.ReLU(),
            nn.Linear(64, 4),
            nn.ReLU(),
            nn.Linear(4, 5),
            nn.Softmax()
        )
        self.to(device)

    def forward(self, x):
        # Block 1
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = torch.flatten(x, 1)
        output = self.fc(x)
        return output


def train(args, model, train_dataloader, val_dataloader):
    device = args.device

    params = model.parameters()

    optimizer = optim.Adam(params, lr=args.lr, betas=[args.beta1, args.beta2],
                           weight_decay=args.weight_decay)

    criterion = nn.CrossEntropyLoss()
    best_acc = -np.inf

    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=10)

    # wandb.watch(model, log_freq=100)

    for epoch in tqdm(range(args.epochs)):
        print('-' * 10)
        print(f'Epoch {epoch + 1}/{args.epochs}')

        batch_train_loss = []
        batch_train_acc = []

        pbar = tqdm(train_dataloader, total=len(train_dataloader))
        for img, label in pbar:
            train_x = img.to(device)
            label = label.to(device)

            pred = model(train_x)

            loss = criterion(pred, label)
            batch_train_loss.append(loss.item())

            _, pred_class = torch.max(pred, dim=1)
            label_class = torch.nonzero(label == 1, as_tuple=False)[:, 1]
            acc = torch.sum(torch.eq(pred_class, label_class)).item() / len(label_class)
            batch_train_acc.append(acc)

            # wandb.log({"train_loss": loss.item(),
            #            "train_accuracy": acc})

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

                val_pred = model(val_x)

                loss_val = criterion(val_pred, val_label)
                batch_val_loss.append(loss_val.item())

                _, pred_class = torch.max(val_pred, dim=1)
                label_class = torch.nonzero(loss_val == 1, as_tuple=False)[:, 1]
                acc_val = torch.sum(torch.eq(pred_class, label_class)).item() / len(label_class)
                batch_val_acc.append(acc_val)

                # wandb.log({"validation_loss": loss_val.item()
                #            "validation_accuracy": acc_val})

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

        scheduler.step(loss_val)

    print(f'\nBest validation accuracy: {best_acc}')
    print(f'\nBest weights can be found here: {best_weights_path}')
