from ultralytics import YOLO
import cv2
import warnings
import numpy as np
import supervision as sv
import torch
from tqdm.auto import tqdm
import argparse
import os
import time
import re


warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='Saving Paths', add_help=False)

parser.add_argument('--name', default='Come', type=str, help='Point; None; Bad; Good; Stop; Come')
parser.add_argument('--path_images2crop', default=r'data/data_LongRANGE/Come',
                    metavar='DIR', help='path to cropped images')
parser.add_argument('--root_image', default=r'data/data_LongRANGE/Come',
                    metavar='DIR', help='path to images')
parser.add_argument('--root_mask', default=r'data/data_LongRANGE/Come',
                    metavar='DIR', help='path to mask')
parser.add_argument('--root_crop', default=r'data/data_LongRANGE/Come',
                    metavar='DIR', help='path to cropped images')


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

    cv2.imwrite(img_name_save, image)
    cv2.imwrite(mask_name_save, mask)
    cv2.imwrite(crop_name_save, cropimg)


def seg_image(args, model, to_save=False):
    main_root = args.path_images2crop

    for k in range(2, 21):
        father_root = os.path.join(main_root, str(k))
        root_images = os.path.join(father_root, 'image')
        list_images = os.listdir(root_images)
        for i in tqdm(range(len(list_images))):
            im_root = os.path.join(root_images, str(list_images[i]))
            image = cv2.imread(im_root)
            w = image.shape[:2][1]
            h = image.shape[:2][0]
            root_mask = os.path.join(father_root, 'mask')
            root_crop = os.path.join(father_root, 'crop')
            if not os.path.exists(root_mask):
                os.makedirs(root_mask)
            if not os.path.exists(root_crop):
                os.makedirs(root_crop)
            mask_path = os.path.join(root_mask, str(list_images[i]))
            crop_path = os.path.join(root_crop, str(list_images[i]))
            # if os.path.exists(crop_path):
            #     pass
            # else:
            results = model(image)[0]
            if results.masks is None:
                mask = np.zeros((h, w, 3), dtype=np.uint8)
                if to_save:
                    cv2.imwrite(mask_path, mask)
                    cv2.imwrite(crop_path, mask)
            else:
                detections = sv.Detections.from_yolov8(results)

                for j in detections.class_id:
                    class_id = model.model.names[j]
                    if class_id == 'person':
                        where = torch.nonzero(results.boxes.cls == float(0))[0][0].item()

                        mask = results.masks.masks[where]

                        mask = mask.detach().cpu().numpy()

                        mask = np.squeeze(mask)
                        mask = cv2.resize(mask, (w, h))

                        cropped = CropingMask(image, mask)
                        mask[np.where(mask > 0)] *= 255

                        if to_save:
                            cv2.imwrite(mask_path, mask)
                            cv2.imwrite(crop_path, cropped)
                    # else:
                    #     mask = np.zeros((h, w, 3), dtype=np.uint8)
                    #     if to_save:
                    #         cv2.imwrite(mask_path, mask)
                    #         cv2.imwrite(crop_path, mask)


def RealTime(args, model, data_size, to_save=False):
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        print("Error: Could not open video device")
    for j in tqdm(range(data_size)):
        ret, frame = cap.read()
        try:
            results = model(frame)[0]

            detections = sv.Detections.from_yolov8(results)

            for i in detections.class_id:
                class_id = model.model.names[i]
                if class_id == 'person':
                    where = torch.nonzero(results.boxes.cls==float(0))[0][0].item()

                    mask = results.masks.masks[where]

                    mask = mask.detach().cpu().numpy()
                    mask = np.squeeze(mask)
                    # mask = cv2.resize(mask, (640, 480))

                    cropped = CropingMask(frame, mask)
                    mask[np.where(mask > 0)] *= 255
                    cv2.imshow('', cropped)

                    if to_save:
                        save_img(args.root_image, args.root_mask, args.root_crop,
                                 args.name, j, frame, mask, cropped)
                    else:
                        continue
        except:
            continue
        if cv2.waitKey(30) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    exit(0)


if __name__ == "__main__":
    # model = YOLO('yolov8n-seg.pt')  # load an official model
    model = YOLO('yolo_pt/yolov8n-seg.pt')  # load a custom model
    data_size = 5000
    args_config = parser.parse_args()
    RealTime(args_config, model, data_size, to_save=False)
    # seg_image(args_config, model, to_save=True)
    # RealTime2(model)
