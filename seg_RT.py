from ultralytics import YOLO
import cv2
import warnings
import numpy as np
import supervision as sv
from tqdm.auto import tqdm
import argparse
import os
import time
import re


warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='Saving Paths', add_help=False)
parser.add_argument('--root_images', default=r'/home/roblab20/PycharmProjects/PDE/data_train', metavar='DIR',
                    help='path to images')
parser.add_argument('--root_mask', default=r'/home/roblab20/PycharmProjects/PDE/data_train', metavar='DIR',
                    help='path to mask')
parser.add_argument('--root_crop', default=r'/home/roblab20/PycharmProjects/PDE/data_train', metavar='DIR',
                    help='path to cropped images')

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
    return cropped_image


def save_img(path_img, path_mask, path_pathcrop,
             name, i, image, mask, cropped_image):
    dim = (640, 480)
    # dim = (1920, 1080)
    dim_crop = (256, 256)
    image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    mask = cv2.resize(mask, dim, interpolation = cv2.INTER_AREA)
    cropimg = cv2.resize(cropped_image, dim_crop, interpolation=cv2.INTER_AREA)

    img_name = path_img + '/' + name + '_{}'.format(i)
    mask_name = path_mask + '/' + name + '_{}'.format(i)
    crop_name = path_pathcrop + '/' + name + '_{}'.format(i)

    tt = str(time.asctime())
    img_name_save = (img_name + " " + str(re.sub('[:!@#$]', '_', tt) + '.png')).replace(' ', '_')
    mask_name_save = (mask_name + " " + str(re.sub('[:!@#$]', '_', tt) + '.png')).replace(' ', '_')
    crop_name_save = (crop_name + " " + str(re.sub('[:!@#$]', '_', tt) + '.png')).replace(' ', '_')

    # cv2.imwrite(img_name_save, image)
    # cv2.imwrite(mask_name_save, mask)
    cv2.imwrite(crop_name_save, cropimg)


def seg_image(model, path_image, path2save):
    image = cv2.imread(path_image)
    w = image.shape[:2][1]
    h = image.shape[:2][0]

    results = model(image)[0]

    mask = results.masks.masks[0]
    mask = mask.detach().cpu().numpy()
    mask = np.squeeze(mask)
    mask = cv2.resize(mask, (w, h))

    cropped = CropingMask(image, mask)

    cropped_name = os.path.join(path2save, path_image)
    # cv2.imwrite(cropped_name, cropped)
    cv2.imshow('', cropped)
    cv2.waitKey(0)


def RealTime(model, data_size, path2save):
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        print("Error: Could not open video device")
    for j in tqdm(range(data_size)):
        ret, frame = cap.read()
        full_path = os.path.join(path2save, str(j))
        try:
            results = model(frame)[0]
            detections = sv.Detections.from_yolov8(results)
            # print(detections)
            for i in detections.class_id:
                class_id = model.model.names[i]
                if class_id == 'person':
                    # print(detections.confidence[i])
                    mask = results.masks.masks[0]

                    mask = mask.detach().cpu().numpy()
                    mask = np.squeeze(mask)
                    mask = cv2.resize(mask, (640, 480))

                    cropped = CropingMask(frame, mask)
                    cv2.imshow('', cropped)
                    cv2.imwrite(f'{full_path}.png', cropped)
        except:
            continue
        if cv2.waitKey(30) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    exit(0)


if __name__ == "__main__":
    image_path = 'NoPoint_13_Wed_Mar__8_15_04_50_2023.png'
    # model = YOLO('yolov8n-seg.pt')  # load an official model
    model = YOLO('yolo_pt/yolov8n-seg.pt')  # load a custom model
    data_size = 100
    path2save = '/home/roblab20/PycharmProjects/LongRange/data/yolo'
    # RealTime(model, data_size, path2save)
    seg_image(model, image_path, path2save)
