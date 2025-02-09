import cv2
import os
import time
import re
import pickle
from tqdm.auto import tqdm


def save_img(path_img, image_width, image_height, name, i, image):
    dim = (image_width, image_height)
    image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

    img_name = path_img + '/' + name + '_{}'.format(i)

    tt = str(time.asctime())
    img_name_save = (img_name + " " + str(re.sub('[:!@#$]', '_', tt) + '.png')).replace(' ', '_')
    cv2.imwrite(img_name_save, image)
    id_name = img_name_save.split('/')[-1]
    return id_name


def save_pkl(data, save_path, distance, data_name='data_record'):
    t = str(time.asctime())
    file_name = data_name + " " + str(distance) + "m" + " " + str(re.sub('[:!@#$]', '_', t) + '.pkl')
    file_name = file_name.replace(' ', '_')
    completeName = os.path.join(save_path, file_name)
    with open(completeName, "wb") as f:
        pickle.dump(data, f)
        del data


def main(name, image_width, image_height, path, pkl_path, data_size, distance, to_save=False):
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, image_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, image_height)
    start = 0
    if not cap.isOpened():
        print("Error: Could not open video device")

    ids_list = []
    for j in tqdm(range(data_size)):
        ret, frame = cap.read()
        cv2.imshow('Acquisition', frame)
        if to_save:
            if start >= 10:
                id_name = save_img(path, image_width, image_height, name, j, frame)
                ids_list.append(id_name)
        start += 1
        if cv2.waitKey(30) == 27:
            break
    if to_save:
        save_dict = {'distance': distance,
                     'id_name': ids_list}
        save_pkl(save_dict, pkl_path, distance)
    cap.release()
    cv2.destroyAllWindows()
    exit(0)


if __name__ == "__main__":
    # Change these parameters according to your requirements\preferences:
    # =================================================================
    parent_dir = r'data_LongRANGE/'  # data_LongRANGE/
    name = r'Stop'  # 'Bad', 'Come', 'Good', 'None', 'Point', 'Stop',
    image_width = 512
    image_height = 512
    data_size = 510
    distance = 25  # In meters
    to_save = True
    # to_save = False
    # =================================================================

    images_path = parent_dir + name + '/' + str(distance) + '/image'
    pkl_path = parent_dir + name + '/' + str(distance) + '/pkl'

    main(name, image_width, image_height, images_path, pkl_path, data_size, distance, to_save=to_save)
