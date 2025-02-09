import cv2
import numpy as np
import os
from PIL import Image, ImageFilter,ImageEnhance


def crop_image(image, x, y, w, h):
    hypotenuse = np.sqrt(h ** 2 + w ** 2)
    num_pixels = int(hypotenuse/6.5)
    cropped_image = image[y - num_pixels:y + h + num_pixels, x - num_pixels - 2:x + w + num_pixels + 2]
    return cropped_image


def object(net, output_layers, ):
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
        object_image = crop_image(image, x, y, w, h)
        # Extract the object from the frame
        # object_image = image[y:y + h, x:x + w]
        if not object_image.size > 0:
            object_image = image[y:y+h, x:x+w]
            if not object_image.size > 0:
                object_image = image

        pil_image = Image.fromarray(object_image)
        b, g, r = pil_image.split()
        im = Image.merge("RGB", (r, g, b))
        sunset_resized = im.resize((512, 512))
        return sunset_resized


def object_edge(net, output_layers, image):
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
        object_image = crop_image(image, x, y, w, h)
        # object_image = image[y-35:y +35 + h, x-35:x +35 + w]
        if not object_image.size > 0:
            object_image = image[y:y+h, x:x+w]
            if not object_image.size > 0:
                object_image = image

        pil_image = Image.fromarray(object_image)
        b, g, r = pil_image.split()
        im = Image.merge("RGB", (r, g, b))
        sunset_resized = im.resize((512, 512))
        # enhancer = ImageEnhance.Sharpness(sunset_resized)
        # sunset_resized = enhancer.enhance(1.5)
        # gray_image = sunset_resized.convert("L")
        # gray_image = gray_image.filter(ImageFilter.FIND_EDGES)
        return sunset_resized


def image_degradation(image, blur_kernel_size, save_path, save=False):
    im = np.array(image)
    degradated_image = cv2.GaussianBlur(im, (3, 3), 2.3)
    # degradated_image = cv2.blur(im, (4, 4))
    block_size = 4
    degradated_image = cv2.resize(degradated_image, (im.shape[1] // block_size, im.shape[0] // block_size))
    degradated_image = cv2.resize(degradated_image, (im.shape[1], im.shape[0]))
    # degradated_image = cv2.GaussianBlur(degradated_image, (3, 3), sigmaX=10, sigmaY=10)
    # degradated_image = cv2.blur(degradated_image, (4, 4))
    degradated_image = cv2.GaussianBlur(degradated_image, (7, 7), sigmaX=2, sigmaY=2)
    # blurred_image = cv2.blur(degradated_image, (4, 4))
    # sharpening_kernel = np.array([[-1, -1, -1],
    #                               [-1, 9, -1],
    #                               [-1, -1, -1]])
    # degradated_image = cv2.filter2D(degradated_image, -1, sharpening_kernel)

    pil_image = Image.fromarray(degradated_image)
    enhancer = ImageEnhance.Sharpness(pil_image)
    pil_image = enhancer.enhance(5)

    if save:
        pil_image.save(save_path, dpi=(pil_image.info.get('dpi', (512, 512))))
    return pil_image


if __name__ == '__main__':
    image_root = '/home/roblab20/PycharmProjects/LongRange/data/for_paper/originals/1'  # Bad, Come, Good, None, Point, Stop
    save_low_root = '/home/roblab20/PycharmProjects/LongRange/data/for_paper/after_crop'
    save_high_root = ''
    blur_kernel_size = 3

    net = cv2.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights')
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    for i, file in enumerate(os.listdir(image_root)):
        print(i)
        image_path = os.path.join(image_root, file)
        # filename = file.split("/")[-1]
        save_low_path = os.path.join(save_low_root, file)
        save_high_path = os.path.join(save_high_root, file)
        # if os.path.exists(save_high_path):
        #     continue
        # else:
        #     image = cv2.imread(image_path)
        #     # print(file)
        #     cropped_image = object(net, output_layers)
        #     if not cropped_image:
        #         continue
        #     else:
        #         degrated_image = image_degradation(cropped_image, blur_kernel_size, save_low_path,  save=False)
        #         # image = Image.fromarray(image)
        #         degrated_image.show()
        #         # cropped_image.save(save_high_path, dpi=(image.info.get('dpi', (512, 512))))

        image = cv2.imread(image_path)
        cropped_image = object_edge(net, output_layers, image)
        cropped_image.save(save_low_path)
        cropped_image.show()


