import torch
import numpy as np
import cv2
from Gesture_class import CNN
from ultralytics import YOLO
from pretrained_models import load_model4test
from utils import CropingMask, yolo_maskNcrop
from torchvision import transforms


def realtime_gesture_class(model, transform, gestures_dict, crop=True):
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    font = cv2.FONT_HERSHEY_SIMPLEX
    if not cap.isOpened():
        print("Error: Could not open video device")
    while True:
        ret, frame = cap.read()
        if crop:
            cropped = yolo_maskNcrop(frame, model)
            input = transform(cropped)
        else:
            input = frame
        with torch.no_grad():
            gesture = model(input)
        _, gesture_class = torch.max(gesture, dim=1)
        gesture_class = gesture_class.item()
        index = list(gestures_dict.values()).index(gesture_class)
        gesture_name = list(gestures_dict.keys())[index]
        cv2.putText(frame,
                    gesture_name,
                    (50, 50),
                    font, 0.5,
                    (255, 255, 255),
                    1,
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

    model_name = ''  # Simple_CNN; DenseNet; EfficientNet; GoogLeNet; VGG; Wide_ResNet
    weights_root = ''

    transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])

    if model_name == 'Simple_CNN':
        model = CNN(device)
        model.load_state_dict(torch.load(weights_root, map_location=device))
        model.eval()
    else:
        model = load_model4test(model_name, num_classes, weights_root)
        model.eval()

    realtime_gesture_class(model, transform, gestures_dict)
