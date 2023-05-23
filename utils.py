import torch
import torchvision
import cv2
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
import supervision as sv


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


def CropingMask(original_image, mask_image):
    imagergb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    color_mask = np.zeros_like(imagergb)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilate = cv2.dilate(mask_image, kernel, iterations=15)
    d = np.array(dilate, dtype=bool)
    color_mask[:, :, 0] += (d * 255).astype('uint8')
    color_mask[:, :, 1] += (d * 255).astype('uint8')
    color_mask[:, :, 2] += (d * 255).astype('uint8')
    res = ((imagergb  / color_mask) * color_mask).astype('uint8')
    res = cv2.cvtColor(res, cv2.COLOR_RGB2BGR)
    positions = np.nonzero(res)
    top = positions[0].min()
    bottom = positions[0].max()
    left = positions[1].min()
    right = positions[1].max()
    cropped_image = res[top:bottom, left:right]
    # cropped_image = cv2.resize(cropped_image, (640, 480))
    return cropped_image


def yolo_maskNcrop(frame, model):
    results = model(frame)[0]

    detections = sv.Detections.from_yolov8(results)

    for i in detections.class_id:
        class_id = model.model.names[i]
        if class_id == 'person':
            where = torch.nonzero(results.boxes.cls == float(0))[0][0].item()

            mask = results.masks.masks[where]

            mask = mask.detach().cpu().numpy()
            mask = np.squeeze(mask)
            mask = cv2.resize(mask, (640, 480))

            cropped = CropingMask(frame, mask)
            mask[np.where(mask > 0)] *= 255
            return cropped

