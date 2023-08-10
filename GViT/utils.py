import torch
from torch import nn
import cv2
import os
import time
import datetime
import subprocess
import re
import torch
import torchvision
from tqdm import tqdm
import numpy as np


def get_index(i, j, image_width):
    return j * image_width + i


def get_idx(image_width, image_height):
    edge_index = []
    for i in range(image_width):
        for j in range(image_height):
            current_index = get_index(i, j, image_width)
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    if 0 <= i + dx < image_width and 0 <= j + dy < image_height:
                        edge_index.append([current_index, get_index(i + dx, j + dy, image_width)])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t()
    return edge_index


def save_net(path, state, epoch):
    tt = str(time.asctime())
    img_name_save = epoch + '_' + 'net' + " " + str(re.sub('[:!@#$]', '_', tt))
    img_name_save = img_name_save.replace(' ', '_') + '.pt'
    _dir = os.path.abspath('../')
    path = os.path.join(_dir, path)
    t = datetime.datetime.now()
    datat = t.strftime('%m/%d/%Y').replace('/', '_')
    dir = os.path.join(path, datat)
    if not os.path.exists(dir):
        try:
            os.makedirs(dir, exist_ok=True)
            print("Directory '%s' created successfully" % ('net' + '_' + datat))
        except OSError as error:
            print("Directory '%s' can not be created" % ('net' + '_' + datat))

    net_path = os.path.join(dir, img_name_save)
    print()
    print(net_path)
    torch.save(state, net_path)
    return net_path


def test(args, model, test_dataloader):
    batch_val_loss = []
    batch_val_acc = []

    edge_index = get_idx(args.img_size, args.img_size).to(args.device)

    criterion = nn.CrossEntropyLoss()

    model.eval()
    with torch.no_grad():
        pbar = tqdm(test_dataloader, total=len(test_dataloader))
        for img, label in pbar:
            val_x = img.to(args.device)
            val_label = label.to(args.device)

            val_pred = model(val_x, edge_index)

            loss_val = criterion(val_pred, val_label)
            batch_val_loss.append(loss_val.item())

            _, pred_class = torch.max(val_pred, dim=1)
            label_class = torch.nonzero(val_label == 1, as_tuple=False)[:, 1]
            acc_val = torch.sum(torch.eq(pred_class, label_class)).item() / len(label_class)
            batch_val_acc.append(acc_val)

            pbar.set_postfix({'Val Loss': loss_val.item(),
                              'Val Acc': acc_val})

            del val_x, val_label, val_pred
            torch.cuda.empty_cache()
    ce = np.mean(batch_val_loss)
    acc = np.mean(batch_val_acc)
    print(f'Test CE: {ce}. Test Accuracy: {acc}')
