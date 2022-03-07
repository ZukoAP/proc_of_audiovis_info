from PIL import Image, ImageDraw, ImageFont
from lab_1.main import binarize, grayscale
from lab_4.main import crop_letter
import scipy.interpolate as interpolate
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
import matplotlib.patches as patches

name = 'str'

img = Image.open(f'./{name}.bmp')
im_array = (np.invert(np.array(img))).astype(int)
# crop_letter(binarize(grayscale(img, True))).save('./str2.bmp')  # initial crop

'''
def profile(img: Image, name):
    w, h = img.size
    image_array = (np.invert(np.array(img))).astype(int)

    projection_y = [sum(row) for row in image_array]

    plt.plot(projection_y, np.arange(0, len(image_array)))
    plt.yticks(np.arange(0, len(image_array), 10.0))
    plt.xticks(np.arange(min(projection_y), max(projection_y) + 1, 30.0))
    plt.gca().invert_yaxis()  # flipping over y axis
    plt.title('horizontal')
    # plt.savefig('./str_horiz.png')
    plt.show()

    projection_x = [sum(row) for row in np.transpose(image_array)]
    # plt.plot(np.arange(0, len(image_array[0])), projection_x)
    plt.xticks(np.arange(0, len(image_array[0]), 50.0))
    plt.yticks(np.arange(min(projection_x), max(projection_x) + 1, 2.0))
    plt.title('vertical')
    fig = plt.figure(figsize=(96, 9.6))
    ax = fig.add_subplot(111)
    ax.plot(np.arange(0, len(image_array[0])), projection_x)
    plt.tight_layout()
    plt.savefig('./str_vert_long.png')
    plt.show()


profile(img, 'str')
'''

vert_hist = np.sum(im_array, axis=0)
print(len(vert_hist))
plt.bar(list(range(0, len(vert_hist))), vert_hist)
plt.yticks(np.arange(0, max(vert_hist), 5.0))
plt.title('normal vertical')
plt.savefig(f'./{name}_vert.png')
plt.close()

horiz_hist = im_array.shape[1] - np.sum(img, axis=1)
print(len(horiz_hist))
plt.barh(list(range(0, len(horiz_hist))), horiz_hist)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.title('normal horizontal')
plt.savefig(f'./{name}_horiz.png')
plt.close()

def segmentize(img_array, name):
    horizontal_hist = img_array.shape[1] - np.sum(img, axis=1)
    h, w = img_array.shape
    position = []
    str_rows = []
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
        position.append([max(0, w_start),  w])
        wend = 0
    print(len(position))
    letter_boxes = []
    for p in position:
        letter_boxes.append(patches.Rectangle((p[0], 0), p[1]-p[0], h, fill=False, linewidth=1, edgecolor='g', facecolor='none'))
        print(p)
    # Create figure and axes
    fig, ax = plt.subplots()
    ax.imshow(img)
    # Add the patch to the Axes
    ax.add_collection(PatchCollection(letter_boxes, True))
    plt.tight_layout()
    plt.savefig(f'./{name}_segmentized.png')
    plt.show()

'''
def segmentize_spec(img_array, name):
    horizontal_hist = img_array.shape[1] - np.sum(img, axis=1)
    h, w = img_array.shape
    position = []
    str_rows = []
    mean = np.mean(horizontal_hist)
    for i in range(len(horizontal_hist)):
        if horizontal_hist[i] > mean * 1.15:
            str_rows.append(i)
    height_up, height_down = str_rows[0]+4, str_rows[-1]
    wstart, wend, w_start, w_end = 0, 0, 0, 0
    img_array = img_array[height_up:height_down]
    vertical_hist = np.sum(img_array, axis=0)
    vert_len = len(vertical_hist)
    # print(vert_len)
    # plt.bar(list(range(0, vert_len)), vertical_hist)
    # plt.yticks(np.arange(0, max(vertical_hist), 1.0))
    # plt.title('cropped vertical')
    # plt.savefig(f'./{name}_vert_cropped.png')
    # plt.close()
    for i in range(vert_len):
        if np.mean(vertical_hist[max(0, i-4):min(i+2, vert_len-1)]) > 7 and wstart == 0:
            w_start = i
            wstart = 1
            wend = 0
        if ((np.mean(vertical_hist[max(0, i-1):min(i+1, vert_len-1)]) <= 2) or vertical_hist[i] == 0) and wstart == 1:
            w_end = i
            wstart = 0
            wend = 1
        # Save coordinates when start and end points are confirmed
        if wend == 1:
            position.append([max(0,w_start-2), min(w_end+2,w)])
            wend = 0
    print(len(position))
    letter_boxes = []
    for p in position:
        letter_boxes.append(patches.Rectangle((p[0], height_up), p[1]-p[0], height_down-height_up, fill=False, linewidth=1, edgecolor='r', facecolor='none'))
        print(p)
    # Create figure and axes
    fig, ax = plt.subplots()
    ax.imshow(img)
    # Add the patch to the Axes
    ax.add_collection(PatchCollection(letter_boxes, True))
    plt.savefig(f'./{name}_segmentized_spec.png')
    plt.show()
'''

segmentize(im_array, name)
