from __future__ import print_function
import time
from tqdm import tqdm
from torchvision import transforms
import cv2
import os
import numpy as np
import torch
from Hourglass.Hourglass_SR import Hourglass
from PIL import Image, ImageEnhance, ImageFilter
import torch.nn as nn
from pretrained_models import load_model4test


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
        return image
    else:
        max_confidence_idx = np.argmax(confidences)
        box = boxes[max_confidence_idx]
        x, y, w, h = box
        hypotenuse = np.sqrt(h**2 + w**2)

        object_image = image[y - 35:y + h + 35, x - 35:x + w + 35]
        if not object_image.size > 0:
            object_image = image[y:y + h, x:x + w]
            if not object_image.size > 0:
                object_image = image

        pil_image = Image.fromarray(object_image)
        b, g, r = pil_image.split()
        im = Image.merge("RGB", (r, g, b))
        sunset_resized = im.resize((512, 512), Image.BILINEAR)

        if hypotenuse < 230:
            sunset_resized = sunset_resized.filter(ImageFilter.GaussianBlur(radius=2))
            sunset_resized = super_resolution(sunset_resized, sr_transform, sr_model, device)
        else:
            enhancer = ImageEnhance.Sharpness(sunset_resized)
            sunset_resized = enhancer.enhance(1.5)

        return sunset_resized


def realtime_gesture_class(input, model, transform):
    gestures_dict = {'None': 0, 'Point': 1, 'Bad': 2, 'Good': 3, 'Stop': 4, 'Come': 5}
    if input is None:
        gesture_name = "None"
        conf = '1'
        gesture_arr = '---'
    else:
        input = transform(input)[None, :].to(device)
        with torch.no_grad():
            gesture = model(input)
        conf, gesture_class = torch.max(gesture, dim=1)
        # gesture_arr = gesture.detach().cpu().numpy()
        conf = "{:.2f}".format(conf.item())
        gesture_class = gesture_class.item()
        index = list(gestures_dict.values()).index(gesture_class)
        gesture_name = list(gestures_dict.keys())[index]
    return gesture_name, conf


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

# CLASSIFIER
# -----------
num_classes = 6
model_class_name = 'DenseNet'  # Simple_CNN; DenseNet; EfficientNet; GoogLeNet; VGG; Wide_ResNet
weights_class_root = '/home/roblab20/PycharmProjects/LongRange/checkpoint/DenseNet/SR_images/no_finetune/07_26_2023/10_net_Wed_Jul_26_14_58_42_2023.pt'
transform_class = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

model_class = load_model4test(model_class_name, num_classes, weights_class_root, device)
model_class.eval()

# OPEN CAMERA AND START CLASSIFYING
# -------------
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
font = cv2.FONT_HERSHEY_SIMPLEX

if not cap.isOpened():
    print("Error: Could not open video device")
while True:
    ret, frame = cap.read()
    frame_zoomed_in = zoom_in(net, output_layers, SR_transform, SR_model, frame, device)
    new_frame = cv2.cvtColor(np.array(frame_zoomed_in), cv2.COLOR_RGB2BGR)
    gesture_name, conf = realtime_gesture_class(frame_zoomed_in, model_class, transform_class)

    cv2.putText(new_frame,
                gesture_name + " " + conf,
                (50, 70),
                font, 2.4,
                # (253, 46, 62),
                (0, 0, 255),
                3,
                cv2.LINE_AA)
    cv2.imshow('Classify', new_frame)
    if cv2.waitKey(5) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()





