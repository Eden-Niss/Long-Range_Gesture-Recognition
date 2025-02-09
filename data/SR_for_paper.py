from __future__ import print_function
import time
from tqdm import tqdm
from torchvision import transforms
import cv2
import os
import numpy as np
import torch
from Hourglass.Hourglass_SR import Hourglass
from Gesture_class import CNN2
from PIL import Image, ImageEnhance, ImageFilter
import torch.nn as nn
from pretrained_models import load_model4test


def crop_image(image, x, y, w, h):
    hypotenuse = np.sqrt(h ** 2 + w ** 2)
    num_pixels = int(hypotenuse/6.5)
    cropped_image = image[y - num_pixels:y + h + num_pixels, x - num_pixels - 2:x + w + num_pixels + 2]
    # cropped_image = image[y - num_pixels:y + h + num_pixels, x - num_pixels:x + w + num_pixels ]
    return cropped_image


def crop_image_bb_only(image, x, y, w, h):
    hypotenuse = np.sqrt(h ** 2 + w ** 2)
    num_pixels = int(hypotenuse/6.5)
    cropped_image = image[y:y + h, x:x + w]
    return cropped_image


def crop_image_fixed_num_pxs(image, x, y, w, h):
    hypotenuse = np.sqrt(h ** 2 + w ** 2)
    num_pixels = 35
    cropped_image = image[y - num_pixels:y + h + num_pixels, x - num_pixels - 2:x + w + num_pixels + 2]
    return cropped_image


def super_resolution(input, sr_transform, sr_model, device):
    if input is None:
        pass
    else:
        frame_T = sr_transform(input).unsqueeze(0)
        with torch.no_grad():
            sr_frame = sr_model(frame_T.to(device))

        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        output = sr_frame.squeeze().cpu().numpy()

        sr_frame = output.transpose((1, 2, 0)) * std + mean
        sr_frame = np.clip(sr_frame, 0, 1)
        sr_frame = (sr_frame * 255).astype(np.uint8)
        sr_frame = Image.fromarray(sr_frame)

        enhancer = ImageEnhance.Sharpness(sr_frame)
        sr_frame = enhancer.enhance(1.5)

        return sr_frame


def zoom_in(net, output_layers, sr_transform, sr_model, image, device):
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Get the bounding boxes, confidence scores, and class IDs
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  # Adjust the confidence threshold as desired
                if class_id==0:
                    center_x = int(detection[0] * image.shape[1])
                    center_y = int(detection[1] * image.shape[0])
                    w = int(detection[2] * image.shape[1])
                    h = int(detection[3] * image.shape[0])
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

    # Select the bounding box with the highest confidence
    if not confidences:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image
    else:
        max_confidence_idx = np.argmax(confidences)
        box = boxes[max_confidence_idx]
        x, y, w, h = box
        hypotenuse = np.sqrt(h**2 + w**2)

        object_image = crop_image(image, x, y, w, h)
        if not object_image.size > 0:
            object_image = image[y:y + h, x:x + w]
            if not object_image.size > 0:
                object_image = image

        pil_image = Image.fromarray(object_image)
        b, g, r = pil_image.split()
        im = Image.merge("RGB", (r, g, b))
        sunset_resized = im.resize((512, 512), Image.BILINEAR)

        # if hypotenuse < 230:
        #     sunset_resized = sunset_resized.filter(ImageFilter.GaussianBlur(radius=2))
        #     sunset_resized = super_resolution(sunset_resized, sr_transform, sr_model, device)
        # else:
        #     enhancer = ImageEnhance.Sharpness(sunset_resized)
        #     sunset_resized = enhancer.enhance(1.5)

        return sunset_resized
        # return im


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# YOLO
# -------
data_dir = '/home/roblab20/PycharmProjects/LongRange/data/data_LongRANGE'
yolo_conf = '/home/roblab20/PycharmProjects/LongRange/yolo_pt/yolov3.cfg'
yolo_weights = '/home/roblab20/PycharmProjects/LongRange/yolo_pt/yolov3.weights'

net = cv2.dnn.readNetFromDarknet(yolo_conf, yolo_weights)
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# SUPER RESOLUTION
# ----------------
SR_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
SR_model = Hourglass()
SR_model = nn.DataParallel(SR_model)

wights_SR_path = '/home/roblab20/PycharmProjects/LongRange/Hourglass/checkpoint/4_net_Thu_Jul__6_17_08_42_2023.pt'
SR_model.load_state_dict(torch.load(wights_SR_path, map_location=device))

SR_model.to(device)
SR_model.eval()

image_path = '/home/roblab20/PycharmProjects/LongRange/data/for_paper/originals/1/Come_11_Sun_Sep__3_18_47_12_2023.png'
path2save = '/home/roblab20/PycharmProjects/LongRange/data/for_paper/after_crop/another'

image = cv2.imread(image_path)

hr_image = zoom_in(net, output_layers, SR_transform, SR_model, image, device)
# hr_image.show()
crop_type = "_new_crop"
name_im = "Come_11_Sun_Sep__3_18_47_12_2023" + crop_type + ".png"
hr_image.save(os.path.join(path2save, name_im))
