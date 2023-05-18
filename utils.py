import torch
import torchvision
import os
import time
import datetime
import subprocess
import re
import torch
import torchvision
from tqdm import tqdm
import torch.nn as nn
import numpy as np


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

    criterion = nn.CrossEntropyLoss()

    model.eval()
    with torch.no_grad():
        pbar = tqdm(test_dataloader, total=len(test_dataloader))
        for img, label in pbar:
            val_x = img.to(args.device)
            val_label = label.to(args.device)

            val_pred = model(val_x)

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
