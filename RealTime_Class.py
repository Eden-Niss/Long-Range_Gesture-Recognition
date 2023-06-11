import torch
import numpy as np
import cv2
from Gesture_class import CNN
from ultralytics import YOLO
from pretrained_models import load_model4test
from utils import CropingMask, yolo_maskNcrop
from torchvision import transforms


def realtime_gesture_class(model, yolo_model, transform, gestures_dict, crop=True):
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    font = cv2.FONT_HERSHEY_SIMPLEX
    if not cap.isOpened():
        print("Error: Could not open video device")
    while True:
        ret, frame = cap.read()
        if crop:
            cropped = yolo_maskNcrop(frame, yolo_model)
            input = transform(cropped)[None, :].to(device)
        else:
            input = frame
            input = transform(input).to(device)
        with torch.no_grad():
            gesture = model(input)
        _, gesture_class = torch.max(gesture, dim=1)
        gesture_class = gesture_class.item()
        index = list(gestures_dict.values()).index(gesture_class)
        gesture_name = list(gestures_dict.keys())[index]
        cv2.putText(frame,
                    gesture_name,
                    (50, 60),
                    font, 3,
                    (0, 0, 0),
                    3,
                    cv2.LINE_AA)
        cv2.imshow('Classify', frame)
        if cv2.waitKey(5) & 0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    num_classes = 6
    gestures_dict = {'None': 0, 'Point': 1, 'Bad': 2, 'Good': 3, 'Stop': 4, 'Come': 5}

    # Classifying
    model_name = 'Simple_CNN'  # Simple_CNN; DenseNet; EfficientNet; GoogLeNet; VGG; Wide_ResNet
    weights_root = ' /home/roblab20/PycharmProjects/LongRange/checkpoint/Simple_cnn/crop/17_net_Wed_May_24_16_18_56_2023.pt'

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # YOLO
    yolo_model = YOLO('yolo_pt/yolov8n-seg.pt')

    if model_name == 'Simple_CNN':
        model = CNN(device)
        model.load_state_dict(torch.load(weights_root, map_location=device))
        model.eval()
    else:
        model = load_model4test(model_name, num_classes, weights_root, device)
        model.eval()

    realtime_gesture_class(model, yolo_model, transform, gestures_dict)
