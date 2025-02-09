import cv2
import numpy as np
import torch
from Hourglass.Hourglass_SR import Hourglass
import torch.nn as nn
import os
from PIL import Image, ImageEnhance, ImageFilter
import torchvision.transforms as transforms
from tqdm import tqdm


def zoom_in(net, output_layers, image, to_be_enhanced):
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
        pass
    else:
        max_confidence_idx = np.argmax(confidences)
        box = boxes[max_confidence_idx]
        x, y, w, h = box
        # print(box)

        # Extract the object from the frame
        # object_image = image[y-20:y+h+20, x-20:x+w+20]
        object_image = image[y - 35:y + h + 35, x - 35:x + w + 35]
        if not object_image.size > 0:
            object_image = image[y:y+h, x:x+w]
            if not object_image.size > 0:
                object_image = image

        pil_image = Image.fromarray(object_image)
        b, g, r = pil_image.split()
        im = Image.merge("RGB", (r, g, b))
        sunset_resized = im.resize((512, 512), Image.BILINEAR)
        # enhancer = ImageEnhance.Sharpness(sunset_resized)
        # sunset_resized = enhancer.enhance(1.5)
        if to_be_enhanced:
            sunset_resized = sunset_resized.filter(ImageFilter.GaussianBlur(radius=2))
        else:
            enhancer = ImageEnhance.Sharpness(sunset_resized)
            sunset_resized = enhancer.enhance(1.5)

        return sunset_resized


def sr_data(input, yolo_cfg_path, yolo_weights, sr_transform, sr_model, to_be_enhanced=True):

    net = cv2.dnn.readNetFromDarknet(yolo_cfg_path, yolo_weights)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    zoom_in_frame = zoom_in(net, output_layers, input, to_be_enhanced)
    if zoom_in_frame is None:
        pass
    else:
        if to_be_enhanced:
            frame_T = sr_transform(zoom_in_frame).unsqueeze(0)
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
        else:
            return zoom_in_frame



yolo_conf = '/home/roblab20/PycharmProjects/LongRange/yolo_pt/yolov3.cfg'
yolo_weights = '/home/roblab20/PycharmProjects/LongRange/yolo_pt/yolov3.weights'


hg_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

hg_model = Hourglass()
hg_model = nn.DataParallel(hg_model)

wights_path = '/home/roblab20/PycharmProjects/LongRange/Hourglass/checkpoint/4_net_Thu_Jul__6_17_08_42_2023.pt'
hg_model.load_state_dict(torch.load(wights_path, map_location=device))

hg_model.to(device)
hg_model.eval()

data_dir = '/home/roblab20/PycharmProjects/LongRange/data/data_LongRANGE/Bad'
SR_data_dir = '/home/roblab20/PycharmProjects/LongRange/data/data_LongRANGE_SR/Bad'

for i in os.listdir(data_dir):
    distance = int(i)
    if distance == 0 or distance >= 8:
        original_dir = os.path.join(data_dir, str(i))
        new_root = os.path.join(SR_data_dir, str(i))
        if not os.path.exists(new_root):
            os.mkdir(new_root)
        new_dir = os.path.join(new_root,  'image')
        if not os.path.exists(new_dir):
            os.mkdir(new_dir)

        for root, directories, files in os.walk(original_dir):
            if root.endswith('image'):
                for filename in tqdm(files):
                    image_path = os.path.join(root, filename)
                    SR_image_path = os.path.join(new_dir, filename)
                    if not os.path.exists(SR_image_path):
                        image = cv2.imread(image_path)
                        hr_image = sr_data(image, yolo_conf, yolo_weights, hg_transform, hg_model)
                        hr_image.show()
                        # if hr_image is not None:
                        #     hr_image.save(SR_image_path)
#
#     if 2 < distance < 8:
#         original_dir = os.path.join(data_dir, str(i))
#         new_root = os.path.join(SR_data_dir, str(i))
#         if not os.path.exists(new_root):
#             os.mkdir(new_root)
#         new_dir = os.path.join(new_root,  'image')
#         if not os.path.exists(new_dir):
#             os.mkdir(new_dir)
#
#         for root, directories, files in os.walk(original_dir):
#             if root.endswith('image'):
#                 for filename in tqdm(files):
#                     image_path = os.path.join(root, filename)
#                     SR_image_path = os.path.join(new_dir, filename)
#                     if not os.path.exists(SR_image_path):
#                         image = cv2.imread(image_path)
#                         crop_image = sr_data(image, yolo_conf, yolo_weights, hg_transform, hg_model, to_be_enhanced=False)
#                         crop_image.show()
#                         print()
#                         # if crop_image is not None:
#                         #     crop_image.save(SR_image_path)
