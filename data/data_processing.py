import cv2
import numpy as np
import os


def image_degradation(image, blur_kernel_size, save_path, save=False):
    degradated_image = cv2.resize(image, (120, 120))
    degradated_image = cv2.resize(degradated_image, (image.shape[1], image.shape[0]))

    smoothed_image = cv2.GaussianBlur(degradated_image, (blur_kernel_size, blur_kernel_size), 0)

    if save:
        cv2.imwrite(save_path, smoothed_image)


if __name__ == '__main__':
    father_root = '/home/roblab20/PycharmProjects/LongRange/data/data_LongRANGE/Bad'
    image_root = father_root + '/4/image'
    save_root = father_root + '/degradated_4/image'
    blur_kernel_size = 5
    for i, file in enumerate(os.listdir(image_root)):
        image_path = os.path.join(image_root, file)
        filename = file.split(".")[0]
        filename = filename + ".jpg"
        save_path = os.path.join(save_root, filename)
        image = cv2.imread(image_path)

        image_degradation(image, blur_kernel_size, save_path,  save=True)


