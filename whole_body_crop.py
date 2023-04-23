import cv2
import mediapipe as mp
import numpy as np
import time
from tqdm.auto import tqdm
import re

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_holistic = mp.solutions.holistic

drawing_thickness = 9
DS = 4
arm_connections = [[12, 14], [14, 16]]
body_quad = [11, 23, 24, 12]
neck_quad = [9, 10, 12, 11]
face = [8, 6, 3, 7, 9, 10]
hand_triangle_connections = [0, 5, 9, 13, 17]
arm_ratio = 5. / 26.  # ratio between arm width and length
hand_ratio = 0.5 / 2.  # Ration between finger segment  width and length
WHITE_COLOR = (255, 255, 255)


def circle_segmentation(image, start, end, ratio=arm_ratio, color=WHITE_COLOR, drawing_thickness=drawing_thickness):
    if start is None or end is None:
        return

    start, end = np.array(start), np.array(end)

    d = np.linalg.norm(np.array(start) - np.array(end))
    radius = int(d * ratio)
    S = np.linspace(0, 1, int(d / DS))
    for s in S:
        px = list(np.array(start) + s * (np.array(end) - np.array(start)))
        px[0], px[1] = int(px[0]), int(px[1])
        cv2.circle(image, px, radius, color, drawing_thickness)


def draw_hand(image, maskImage, landmark_list, connections, arm=False):
    image_rows, image_cols, _ = image.shape
    connect = mp_holistic.HAND_CONNECTIONS


    # if not landmark_list or not connections:
    #     return

    PX = [] # List of the whole 21 hands landmarks' locations at pixel's coordinates
    for idx, landmark in enumerate(landmark_list.landmark):
        landmark_px = mp_drawing._normalized_to_pixel_coordinates(landmark.x, landmark.y, image_cols, image_rows)
        PX.append(landmark_px)

    # Draws the connections if the start and end landmarks are both visible.
    # connections == "hand_triangle_connections"
    for i, connection in enumerate(connect):
        start_idx = connection[0]
        end_idx = connection[1]
        if np.any(i == np.array([1, 3])):
            circle_segmentation(maskImage, PX[start_idx], PX[end_idx], ratio=hand_ratio * 0.4)
        elif np.any(i == np.array([8, 5, 13])):
            continue
        elif np.any(i == np.array([10, 7, 11, 2])):
            circle_segmentation(maskImage, PX[start_idx], PX[end_idx], ratio=hand_ratio * 0.5)
        else:
            circle_segmentation(maskImage, PX[start_idx], PX[end_idx], ratio=hand_ratio)
    pts = np.array([PX[i] for i in hand_triangle_connections], np.int32)
    cv2.fillPoly(maskImage, [pts], WHITE_COLOR)

    # If arm is missing, try to approximate forearm
    if arm:
        px = list(np.array(PX[9]) + 2.5 * (np.array(PX[0]) - np.array(PX[9])))
        px[0], px[1] = int(px[0]), int(px[1])
        circle_segmentation(maskImage, PX[0], px, ratio=int(9. / 26.))


def draw_arm(image, maskImage, landmark_list, connections, neck, face):
    image_rows, image_cols, _ = image.shape
    connect = mp_holistic.POSE_CONNECTIONS
    # if not landmark_list or not connections:
    #     return

    PX = []  # List of the whole 33 body landmarks' locations at pixel's coordinates
    for idx, landmark in enumerate(landmark_list.landmark):
        landmark_px = mp_drawing._normalized_to_pixel_coordinates(landmark.x, landmark.y, image_cols, image_rows)
        PX.append(landmark_px)

    pts = np.array([PX[i] for i in connections], np.int32)
    cv2.fillPoly(maskImage, [pts], WHITE_COLOR)

    pts2 = np.array([PX[i] for i in neck], np.int32)
    cv2.fillPoly(maskImage, [pts2], WHITE_COLOR)

    cv2.circle(maskImage, PX[0], 40, WHITE_COLOR, -1)

    # Draws the connections if the start and end landmarks are both visible.
    # connections == "arm_connections"
    for i, connection in enumerate(connect):
        if len(connection) == 2:
            start_idx = connection[0]
            end_idx = connection[1]
            circle_segmentation(maskImage, PX[start_idx], PX[end_idx], ratio=arm_ratio)


def countdown_start(time_factor_start):
    print(" ")
    print("The recording will begin in:")
    for i in range(0, 3):
        print(3 - i)
        time.sleep(time_factor_start)
    print(" ")


def countdown(time_factor):
    print(" ")
    print("take a frame:")
    time.sleep(time_factor)
    print(" ")


def save_img(path_img, path_mask, path_pathcrop,
             name, i, image, mask, cropped_image):
    dim = (640, 480)
    # dim = (1920, 1080)
    dim_crop = (256, 256)
    image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    mask = cv2.resize(mask, dim, interpolation=cv2.INTER_AREA)
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


def getcoloredMask(image, mask):
    color_mask = np.zeros_like(image)
    color_mask[:, :, 1] += (mask * 250).astype('uint8')
    masked = cv2.addWeighted(image, 1, color_mask, 1.0, 0.0)
    return masked


def CropingMask(original_image, mask_image):
    imagergb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    color_mask = np.zeros_like(imagergb)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilate = cv2.dilate(mask_image, kernel, iterations=5)
    d = np.array(dilate, dtype=bool)
    color_mask[:, :, 0] += (d * 255).astype('uint8')
    color_mask[:, :, 1] += (d * 255).astype('uint8')
    color_mask[:, :, 2] += (d * 255).astype('uint8')
    res = ((imagergb / color_mask) * color_mask).astype('uint8')
    res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
    positions = np.nonzero(res)
    top = positions[0].min()
    bottom = positions[0].max()
    left = positions[1].min()
    right = positions[1].max()
    # output = cv2.rectangle(cv2.cvtColor(res, cv2.COLOR_GRAY2BGR)
    #                        , (left, top), (right, bottom), (0, 255, 0), 1)
    cropped_image = res[top:bottom, left:right]
    return cropped_image


if __name__ == '__main__':
    # For webcam input:
    start = 0.0
    cap = cv2.VideoCapture(0)
    W, H = 480, 640
    # W, H = 1920, 1080
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cap.set(cv2.CAP_PROP_FPS, 30)
    data_points_to_record = 100

    class_img = 0
    if class_img == 1:
        save_path = r'/home/roblab20/PycharmProjects/PointClassification/eran_seg/data/more_data/19_01/Point/image'
        save_pathmask = r'/home/roblab20/PycharmProjects/PointClassification/eran_seg/data/10_01/Point'
        save_pathcrop = r'/home/roblab20/PycharmProjects/LongRange/data/3_4/crop'
        name = 'Point'
    else:
        save_path = r'/home/roblab20/PycharmProjects/LongRange/data/3_4/image'
        save_pathmask = r'/home/roblab20/PycharmProjects/PointClassification/eran_seg/data/04_01/mask/NoPoint'
        save_pathcrop = r'/home/roblab20/PycharmProjects/LongRange/data/3_4/crop'

        name = 'NoPoint'

    # countdown_start(2.2)
    countdown_start(1)
    with mp_hands.Hands(
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:
        for i in tqdm(range(data_points_to_record)):
            success, image = cap.read()

            if not success:
                print("Ignoring empty camera frame.")
                continue

            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            maskImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # fps = int(cap.get(cv2.CAP_PROP_FPS))
            maskImage[:, :] = 0
            results_hand = hands.process(image)
            with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
                results_pose = pose.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            try:
                if results_pose.pose_landmarks:
                    pose_landmarks = results_pose.pose_landmarks
                    draw_arm(image, maskImage, pose_landmarks, body_quad, neck_quad, face)
                else:
                    print('No body visible!')

                if results_hand.multi_hand_landmarks:
                    for a, hand_landmarks in enumerate(results_hand.multi_hand_landmarks):
                        draw_hand(image, maskImage, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                  arm=not results_pose.pose_landmarks)
                        # if results_hand.multi_handedness[a].classification[0].label == 'Left':


            except:
                pass

            # cv2.imshow('Original image', cv2.flip(image, 1))
            cv2.imshow('Mask image', cv2.flip(maskImage, 1))

            try:
                cropped_image = CropingMask(cv2.flip(image, 1), cv2.flip(maskImage, 1))  # cropped_image = image
                # cv2.imshow('gray', cropped_image)
                if start >= 10:
                    pass
                    save_img(save_path, save_pathmask, save_pathcrop,
                             name, i, cv2.flip(image, 1), cv2.flip(maskImage, 1), cropped_image)
            except:
                pass

            if cv2.waitKey(5) & 0xFF == 27:
                break
            start += 1

    cap.release()
    cv2.destroyAllWindows()
    exit(0)
