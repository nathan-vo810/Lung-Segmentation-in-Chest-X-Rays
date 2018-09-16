import cv2
import numpy as np
import os


def cut_lung(image, width, height, label):
    scale_factor = 5

    image = cv2.resize(image, (0, 0), fx=1 / scale_factor, fy=1 / scale_factor)
    label = cv2.resize(label, (0, 0), fx=1 / scale_factor, fy=1 / scale_factor)

    scaled_h = int(height / scale_factor)
    scaled_w = int(width / scale_factor)

    result = np.zeros((scaled_h, scaled_w, 3))
    for i in range(scaled_h):
        for j in range(scaled_w):
            if label[i, j] > 10:
                result[i, j, :] = image[i, j, :]
            else:
                result[i, j, :] = 255

    result = cv2.resize(result, (width, height))

    return result


def deblur_label(label, width, height):
    for i in range(label.shape[0]):
        for j in range(label.shape[1]):
            if label[i][j] > 20:
                label[i][j] = 255
    label = cv2.resize(label, (width, height), interpolation=cv2.INTER_NEAREST)
    return label


def post_processing(data_dir, image_name, image_shape):
    width, height = image_shape
    width, height = int(width), int(height)

    image = cv2.imread(os.path.join(data_dir, 'test_data', image_name))
    label = cv2.imread(os.path.join(data_dir, 'result', image_name), 0)

    label = deblur_label(label, width, height)
    # processed_image = cut_lung(image, width, height, label)

    return label