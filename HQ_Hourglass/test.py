import torch
from tqdm import tqdm
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from pytorch_msssim import ssim
from PIL import Image
from utils import canny


def test(args, model, vgg_model, test_dataloader):
    batch_val_loss = []

    # mae = nn.L1Loss()
    mse = nn.MSELoss()

    model.eval()
    with torch.no_grad():
        i = 0
        pbar = tqdm(test_dataloader, total=len(test_dataloader))
        for img, label in pbar:
            val_x = img.to(args.device)
            val_label = label.to(args.device)

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

            loss_val = loss1 + loss3 + 0.1 * loss2

            batch_val_loss.append(loss_val.item())

            pbar.set_postfix({'Val Loss': loss_val.item()})

            del val_x, val_label, val_pred
            torch.cuda.empty_cache()
    total_loss = np.mean(batch_val_loss)
    print(f'Test loss: {total_loss}')
