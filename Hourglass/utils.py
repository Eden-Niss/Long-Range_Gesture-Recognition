import os
import time
import datetime
import re
import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image, ImageFilter


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


def canny(batch, device):
    to_tensor = transforms.ToTensor()
    size, *_ = batch.size()
    sobel_batch = []
    for i in range(size):
        img = batch[i].detach().cpu().numpy()
        img = img.transpose((1, 2, 0))
        edges = cv2.Canny(np.uint8(img), threshold1=100, threshold2=200)
        edges = Image.fromarray(edges)
        sobel_batch.append(to_tensor(edges))
    sobel_batch = torch.stack(sobel_batch, dim=0).to(device)
    return sobel_batch


def save_results(input_img, output, epoch, models_name, root_path):
    tt = str(time.asctime())
    img_name_save_input = str(epoch) + '_' + 'input' + '_'  + str(models_name) + " " + str(re.sub('[:!@#$]', '_', tt))
    img_name_save_input = img_name_save_input.replace(' ', '_') + '.png'

    img_name_save_output = str(epoch) + '_' + 'output' + '_' + str(models_name) + " " + str(re.sub('[:!@#$]', '_', tt))
    img_name_save_output = img_name_save_output.replace(' ', '_') + '.png'

    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    input_img = input_img.squeeze().cpu().numpy()
    output = output.squeeze().cpu().numpy()

    hr_img = output.transpose((1, 2, 0)) * std + mean
    hr_img = np.clip(hr_img, 0, 1)
    hr_img = (hr_img * 255).astype(np.uint8)
    hr_img = Image.fromarray(hr_img)

    lr_img = input_img.transpose((1, 2, 0)) * std + mean
    lr_img = np.clip(lr_img, 0, 1)
    lr_img = (lr_img * 255).astype(np.uint8)
    lr_img = Image.fromarray(lr_img)

    before = os.path.join(root_path, img_name_save_input)
    after = os.path.join(root_path, img_name_save_output)

    lr_img.save(before)
    hr_img.save(after)
