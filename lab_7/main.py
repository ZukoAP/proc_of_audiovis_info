import numpy as np
from PIL import Image
import sys
import lab_1.main as sampling

img = Image.open(sys.argv[1])


def vectorized_glcm(image, distance, direction):
    img_arr = np.array(image)

    glcm = np.zeros((256, 256), dtype=int)

    if direction == 1:  # direction 90° up ↑
        for i in range(distance, img_arr.shape[0]):
            for j in range(0, img_arr.shape[1]):
                glcm[img_arr[i, j], img_arr[i - distance, j]] += 1
    elif direction == 2:  # direction 45° up-right ↗
        for i in range(distance, img_arr.shape[0]):
            for j in range(0, img_arr.shape[1] - distance):
                glcm[img_arr[i, j], img_arr[i - distance, j + distance]] += 1
    elif direction == 3:  # direction 0° right →
        for i in range(0, img_arr.shape[0]):
            for j in range(0, img_arr.shape[1] - distance):
                glcm[img_arr[i, j], img_arr[i, j + distance]] += 1
    elif direction == 4:  # direction -45° down-right ↘
        for i in range(0, img_arr.shape[0] - distance):
            for j in range(0, img_arr.shape[1] - distance):
                glcm[img_arr[i, j], img_arr[i + distance, j + distance]] += 1

    return glcm


x = vectorized_glcm(sampling.grayscale(img, True), 2, 1)
print(x)
