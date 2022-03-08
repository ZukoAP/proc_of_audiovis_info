import math
from PIL import Image, ImageDraw, ImageFont
from lab_4.main import get_features, crop_letter
from lab_1.main import binarize, grayscale
import scipy.interpolate as interpolate
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
import matplotlib.patches as patches
import csv

name = 'hello there 36'

image = Image.open(f'./{name}.bmp')

# crop_letter(binarize(grayscale(image, True))).save(f'./{name}.bmp')  # initial crop


def proximity(img: Image):
    img_array = (np.invert(np.array(img))).astype(int)
    h, w = img_array.shape
    position = []
    wstart, wend, w_start, w_end = 0, 0, 0, 0
    vertical_hist = np.sum(img_array, axis=0)
    vert_len = len(vertical_hist)
    for i in range(vert_len):
        if vertical_hist[i] > 0 and wstart == 0:
            w_start = i
            wstart = 1
            wend = 0
        if vertical_hist[i] == 0 and wstart == 1:
            w_end = i
            wstart = 0
            wend = 1
        # Save coordinates when start and end points are confirmed
        if wend == 1:
            position.append([max(0, w_start), min(w_end, w)])
            wend = 0
    if wstart == 1:
        position.append([max(0, w_start), w])
    writer = csv.writer(open(f'./{name}.csv', 'w', newline=''), delimiter=';')
    for p in position:
        proximities = {}
        letter = crop_letter(img.crop((p[0], 0, p[1], h)))
        _, letter_weight_b_norm, _, _, letter_grav_centre_x_norm, letter_grav_centre_y_norm, _, _, letter_inert_x_norm, letter_inert_y_norm = get_features(
            letter)
        reader = csv.reader(open('../lab_4/letters/features.csv'), delimiter=',')
        next(reader)
        for row in reader:
            proximities[row[0]] = round(math.sqrt((round(letter_weight_b_norm, 2) - float(row[2])) ** 2 + (
                    round(letter_grav_centre_x_norm, 2) - float(row[5])) ** 2 + (
                                                                round(letter_grav_centre_y_norm, 2) - float(row[6])) ** 2 + (
                                                                round(letter_inert_x_norm, 2) - float(row[9])) ** 2 + (
                                                                round(letter_inert_y_norm,2) - float(row[10])) ** 2), 2)
        hypotheses = sorted(proximities.items(), key=lambda item: item[1])
        maximum = max(proximities.items(), key=lambda item: item[1])[1]
        hypotheses = [(v[0], round(1 - v[1] / maximum, 2)) for v in hypotheses]
        print(hypotheses[0][0], end=' ')
        writer.writerow(hypotheses)


proximity(image)
