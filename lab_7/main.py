from collections import defaultdict

import numpy as np
from PIL import Image
import sys
import lab_1.main as sampling
import matplotlib.pyplot as plt


def getglcm(image, distance, directions):
    image_arr = np.array(image)
    w, h = image.size
    glcm = np.zeros((256, 256), dtype=int)
    k = 0
    for direction in directions:
        if direction == 90:  # ↑
            first = image_arr[distance:, :]
            second = image_arr[:-distance, :]
            k = 2 * w * h - 2 * distance * w
        elif direction == 45:  # ↗
            first = image_arr[distance:, :-distance]
            second = image_arr[:-distance, distance:]
            k = 2 * w * h - 2 * distance * (w + h) + 2 * distance ** 2
        elif direction == 0:  # ->
            first = image_arr[:, :-distance]
            second = image_arr[:, distance:]
            k = 2 * w * h - 2 * distance * h
        elif direction == -45:  # ↘
            first = image_arr[:-distance, :-distance]
            second = image_arr[distance:, distance:]
            k = 2 * w * h - 2 * distance * (w + h) + 2 * distance ** 2

        for i, j in zip(first.ravel(), second.ravel()):
            glcm[i, j] += 1
    glcm += glcm.T
    return glcm, k


def cumsumhistplot(array, path):
    hist, bins = np.histogram(array, 256, [0, 255])
    cs = hist.cumsum()
    cs_normalized = cs * float(hist.max()) / cs.max()
    plt.plot(cs_normalized)
    # show the histogram
    plt.hist(array, bins=255)
    plt.xlim([0, 255])
    plt.legend(('Cumulative sum', 'histogram'), loc='upper left')
    plt.savefig(f'{path}.png')
    plt.show()


def equalize_histogram(gray_image: Image):
    img_hist = gray_image.histogram()
    num_pixels = np.sum(img_hist)
    img_hist_norm = img_hist / num_pixels
    cumulative_hist = np.cumsum(img_hist_norm)

    transform_map = np.floor(255 * cumulative_hist).astype('uint8')

    gray_arr = np.array(gray_image)
    flat = gray_arr.flatten()

    eq_img_flat = [transform_map[p] for p in flat]

    eq_img = np.reshape(np.asarray(eq_img_flat), gray_arr.shape)

    equ_img = Image.fromarray(eq_img.astype('uint8'), mode='L')
    equ_img.save(sys.argv[1][:sys.argv[1].find(".bmp")] + "_eq" + sys.argv[1][sys.argv[1].find(".bmp"):])

    cumsumhistplot(flat, sys.argv[1][:sys.argv[1].find(".bmp")] + "_hist_orig")
    cumsumhistplot(eq_img_flat, sys.argv[1][:sys.argv[1].find(".bmp")] + "_hist_eq")


def equalize_hist(gray_image):
    img_hist = gray_image.histogram()
    map = {}
    map = defaultdict(lambda: -1, map)
    num_pixels = np.sum(img_hist)
    mean = num_pixels / 256
    equ = np.zeros(256, dtype=int)
    k = 0
    cumsum = img_hist[0]
    print(mean)
    for i in range(1, 256):
        if img_hist[i] < mean:
            if np.abs(cumsum + img_hist[i] - mean) <= np.abs(cumsum - mean):
                cumsum += img_hist[i]
                print(cumsum)
                map[i] = k
                print(i, img_hist[i], k)
            else:
                equ[k] = cumsum
                print(cumsum)
                k += 1
                map[i] = k
                print(i, img_hist[i], k)
                cumsum = img_hist[i]
        else:
            equ[k] = cumsum
            print(cumsum)
            equ[k+1] = img_hist[i]
            # k += 1
            map[i] = k + 1
            print(i, img_hist[i], k + 1)
            k += int(np.round(img_hist[i] / mean))
            cumsum = 0

    gray_arr = np.array(gray_image)
    flat = gray_arr.flatten()
    equ_norm = equ / num_pixels
    cumulative_hist = np.cumsum(equ)
    transform_map = np.floor(255 * cumulative_hist).astype('uint8')
    eq_img_flat = [map[p] for p in flat]

    eq_img = np.reshape(np.asarray(eq_img_flat), gray_arr.shape)

    cumsumhistplot(flat, sys.argv[1][:sys.argv[1].find(".bmp")] + "_hist_orig")
    cumsumhistplot(eq_img_flat,  sys.argv[1][:sys.argv[1].find(".bmp")] + "_hist_eq")



if __name__ == "__main__":
    img = Image.open(sys.argv[1])
    # gray_image = sampling.grayscale(img, True)
    # gray_image.save(sys.argv[1][:sys.argv[1].find(".png")] + ".bmp")
    #                 + sys.argv[1][sys.argv[1].find(".bmp"):])
    #
    # glcm, k = getglcm(gray_image, 2, [0, 90])
    #
    # glcm_img = Image.fromarray((glcm * 255 / np.max(glcm)).astype('uint8'), 'L')
    # glcm_img.save(sys.argv[1][:sys.argv[1].find(".bmp")] + "_glcm" + sys.argv[1][sys.argv[1].find(".bmp"):])
    #
    # norm_glcm = glcm / k
    # contrast = 0
    # for (i, j), p in np.ndenumerate(norm_glcm):
    #     contrast += p * ((i - j) ** 2)
    # print(f"Contrast: {round(contrast, 2)}")
    #
    # homogeneity = 0
    # for (i, j), p in np.ndenumerate(norm_glcm):
    #     homogeneity += p / (1 + ((i - j) ** 2))
    # print(f"Homogeneity: {round(homogeneity, 2)}")

    equalize_histogram(img)
