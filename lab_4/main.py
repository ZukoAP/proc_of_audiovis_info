from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from lab_1.main import binarize
import numpy as np
import scipy.interpolate as interpolate
import os
import csv


def generate_letters(font_path, font_size: int):
    font = ImageFont.truetype(font_path, font_size)
    for i in range(ord('a'), ord('z') + 1):
        w, h = font.getsize(chr(i))
        img = Image.new('L', (2 * w, 2 * h), 'white')
        draw = ImageDraw.Draw(img)
        W, H = draw.textsize(chr(i), font=font)
        draw.text((w, h), chr(i), font=font, fill='black', anchor="mm")
        newpath = f'./letters/{chr(i)}'
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        crop_letter(binarize(img)).save(f'./letters/{chr(i)}/{chr(i)}.bmp')  # average otsu threshold = 137


def crop_letter(letter: Image):
    w, h = letter.size
    flag = True
    for y in range(h):
        for x in range(w):
            if letter.getpixel((x, y)) == 0:
                up = y
                flag = False
                break
        if not flag:
            break
    flag = True
    for y in reversed(range(h)):
        for x in range(w):
            if letter.getpixel((x, y)) == 0:
                down = y
                flag = False
                break
        if not flag:
            break
    flag = True
    for x in range(w):
        for y in range(h):
            if letter.getpixel((x, y)) == 0:
                left = x
                flag = False
                break
        if not flag:
            break
    flag = True
    for x in reversed(range(w)):
        for y in range(h):
            if letter.getpixel((x, y)) == 0:
                right = x
                flag = False
                break
        if not flag:
            break
    return letter.crop((left, up, right + 1, down + 1))


def weight_black(img: Image):
    w, h = img.size
    weight = 0
    for y in range(h):
        for x in range(w):
            if img.getpixel((x, y)) == 0:
                weight += 1
    return weight


def gravity_centre(img: Image):
    w, h = img.size
    centre_x = 0
    centre_y = 0
    for y in range(h):
        for x in range(w):
            if img.getpixel((x, y)) == 0:
                centre_x += x
                centre_y += y
    weight = weight_black(img)
    return centre_x // weight, centre_y // weight


def inertial_moments(img: Image):
    w, h = img.size
    grav_centre_x, grav_centre_y = gravity_centre(img)
    inert_x, inert_y = 0, 0
    for y in range(h):
        for x in range(w):
            if img.getpixel((x, y)) == 0:
                inert_x += (y - grav_centre_y) ** 2
                inert_y += (x - grav_centre_x) ** 2
    return inert_x, inert_y


def get_features(img: Image):
    w, h = img.size
    area = w * h
    weight_b = weight_black(img)
    weight_b_rel = weight_b / area
    weight_w = area - weight_b
    grav_centre_x, grav_centre_y = gravity_centre(img)
    grav_centre_x_rel, grav_centre_y_rel = ((grav_centre_x - 1) / w, (grav_centre_y - 1) / h)
    inert_x, inert_y = inertial_moments(img)
    inert_x_rel, inert_y_rel = (
        (inert_x - 1) / (weight_b ** 2), (inert_y - 1) / (weight_b ** 2))  # Mu'_p,q = Mu_p,q / (Mu_0,0 ^ (1+ (p+q)/2))
    return weight_b, weight_b_rel, grav_centre_x, grav_centre_y, grav_centre_x_rel, grav_centre_y_rel, inert_x, inert_y, inert_x_rel, inert_y_rel


def profile(img: Image, name):
    w, h = img.size
    image_array = (np.invert(np.array(img))).astype(int)

    projection_y = [sum(row) for row in image_array]

    plt.plot(projection_y, np.arange(0, len(image_array)))
    plt.yticks(np.arange(0, len(image_array), 1.0))
    plt.xticks(np.arange(min(projection_y), max(projection_y) + 1, 1.0))
    # flipping over y axis
    plt.gca().invert_yaxis()
    plt.title('horizontal')
    plt.savefig(f'./letters/{name}/{name}_horiz.png')
    plt.close()
    '''
    # interpolation
    y_new = np.linspace(0, len(image_array), 300)
    a_BSpline = interpolate.make_interp_spline(np.arange(0, len(image_array)), projection_y)
    x_new = a_BSpline(y_new)
    plt.plot(x_new, y_new)
    plt.yticks(np.arange(0, len(image_array), 1.0))
    plt.xticks(np.arange(min(projection_y), max(projection_y)+1, 1.0))
    plt.gca().invert_yaxis()
    plt.show()
    '''
    projection_x = [sum(row) for row in np.transpose(image_array)]
    plt.plot(np.arange(0, len(image_array[0])), projection_x)
    plt.xticks(np.arange(0, len(image_array[0]), 1.0))
    plt.yticks(np.arange(min(projection_x), max(projection_x) + 1, 1.0))
    plt.title('vertical')
    plt.savefig(f'./letters/{name}/{name}_vert.png')
    plt.close()


if __name__ == "__main__":
    # generate_letters('C:/Windows/Fonts/timesi.ttf', 52)
    # generate_letters('./elvish ring nfi.ttf', 60)  # Tengwar like english alphabet for simplicity in next lab
    generate_letters('./StandardCelticRuneExtended-Regular.ttf', 60)

    with open('./letters/features.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            ['name', 'black weight', 'black weight normalized', 'centre of gravity x', 'centre of gravity y',
             'centre of gravity x normalized', 'centre of gravity y normalized', 'moment of inertia x',
             'moment of inertia y', 'moment of inertia x normalized', 'moment of inertia y normalized'])
        for i in range(ord('a'), ord('z') + 1):
            img = Image.open(f'./letters/{chr(i)}/{chr(i)}.bmp')
            weight_b, weight_b_rel, grav_centre_x, grav_centre_y, grav_centre_x_rel, grav_centre_y_rel, inert_x, inert_y, inert_x_rel, inert_y_rel = get_features(
                img)
            writer.writerow(
                [chr(i), weight_b, round(weight_b_rel, 2), grav_centre_x, grav_centre_y, round(grav_centre_x_rel, 2),
                 round(grav_centre_y_rel, 2),
                 inert_x, inert_y, round(inert_x_rel, 2), round(inert_y_rel, 2)])
            profile(img, chr(i))
