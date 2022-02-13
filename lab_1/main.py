from PIL import Image
import numpy as np
import sys
from fractions import Fraction

NEAREST = 0
BILINEAR = 1


def upsample(image: Image, factor, sampling_mode=NEAREST):
    if sampling_mode != NEAREST and sampling_mode != BILINEAR:
        sampling_mode = NEAREST
    if factor == 1:
        return image.copy()
    width, height = image.size
    # define new image size
    new_width, new_height = int(np.round(width * factor)), int(np.round(height * factor))
    print(new_width, new_height)
    # create new blank image
    upsampled_image = Image.new(image.mode, (new_width, new_height), 'white')

    for y in range(new_height):
        for x in range(new_width):

            if sampling_mode == NEAREST:
                # take coordinates of nearest neighbour, aka nearest-neighbour interpolation
                x_nearest = int(np.floor(x / factor))
                y_nearest = int(np.floor(y / factor))
                pixel = image.getpixel((x_nearest, y_nearest))
                upsampled_image.putpixel((x, y), pixel)

            elif sampling_mode == BILINEAR:
                # find coordinate of prototype of new pixel
                x_proto = x / factor
                y_proto = y / factor

                # Finding neighboring points in crate
                x1 = min(int(np.floor(x_proto)), width - 1)
                y1 = min(int(np.floor(y_proto)), height - 1)
                x2 = min(x1 + 1, width - 1)
                y2 = min(y1 + 1, height - 1)
                q11 = np.array(image.getpixel((x1, y1)))  # down-left
                q12 = np.array(image.getpixel((x1, y2)))  # up-left
                q21 = np.array(image.getpixel((x2, y1)))  # down-right
                q22 = np.array(image.getpixel((x2, y2)))  # up-right

                # Interpolating P1 and P2
                r1 = (x2 - x_proto) * q11 + (x_proto - x1) * q21
                r2 = (x2 - x_proto) * q12 + (x_proto - x1) * q22

                if x1 == x2:
                    r1 = q11
                    r2 = q22

                # Interpolating pixel
                pixel = (y2 - y_proto) * r1 + (y_proto - y1) * r2

                if y1 == y2:
                    pixel = r1

                # Rounding pixel to an int tuple
                pixel = np.round(pixel)
                pixel = tuple(pixel.astype(int))

                upsampled_image.putpixel((x, y), pixel)

    # Save updsampled image
    return upsampled_image


def downsample(image: Image, factor):
    if factor == 1:
        return image.copy()
    width, height = image.size
    # define new image size
    new_width, new_height = int(np.round(width / factor)), int(np.round(height / factor))
    print(new_width, new_height)
    downsampled_image = Image.new(image.mode, (new_width, new_height), 'white')

    box_width = int(np.ceil(factor))
    box_height = int(np.ceil(factor))

    image_array = np.array(image)

    for y in range(new_height):
        for x in range(new_width):
            x_proto = int(np.floor(x * factor))
            y_proto = int(np.floor(y * factor))

            x_end = min(x_proto + box_width, width - 1)
            y_end = min(y_proto + box_height, height - 1)

            pixel = image_array[y_proto:y_end, x_proto:x_end].mean(axis=(0, 1))

            pixel = np.round(pixel)
            pixel = tuple(pixel.astype(int))

            downsampled_image.putpixel((x, y), pixel)
    return downsampled_image


def resample(image: Image, factor):
    print(factor)
    if not isinstance(factor, int):
        factor = "{:.2f}".format(factor)
    print(factor)
    upsampling_factor, downsampling_factor = Fraction(factor).numerator, Fraction(factor).denominator
    print(f'upsampling by {upsampling_factor}, downsampling by {downsampling_factor}')
    return downsample(upsample(image, upsampling_factor), downsampling_factor)


def resample_(image: Image, factor):
    if factor == 1:
        return image.copy()
    elif factor > 1:
        return upsample(image, factor)
    else:
        return downsample(image, 1 / factor)


img = Image.open(sys.argv[1])

upsample(img, 2).save(sys.argv[1][:sys.argv[1].find(".bmp")] + "_upsampled_nearest" + sys.argv[1][sys.argv[1].find(".bmp"):])

# downsample(img, 2).save(sys.argv[1][:sys.argv[1].find(".bmp")] + "_downsampled_box" + sys.argv[1][sys.argv[1].find(".bmp"):])

_x = 5.12


# resample_(img, _x).save(sys.argv[1][:sys.argv[1].find(".bmp")] + f"_{_x}x_resampled_one_way" + sys.argv[1][sys.argv[1].find(".bmp"):])

def grayscale(image: Image, linear_approximation: bool = False):
    width, height = image.size
    gray_img = Image.new('L', (width, height), 'white')
    for y in range(height):
        for x in range(width):
            orig_pixel = image.getpixel((x, y))
            if linear_approximation:
                pixel = int(
                    np.round(0.299 * orig_pixel[0]) + np.round(0.587 * orig_pixel[1]) + np.round(0.114 * orig_pixel[2]))
            else:
                pixel = np.mean(orig_pixel)
                pixel = int(np.round(pixel))
            gray_img.putpixel((x, y), pixel)
    return gray_img


#grayscale(img, True).save(sys.argv[1][:sys.argv[1].find(".bmp")] + "_gray_linear_approx" + sys.argv[1][sys.argv[1].find(".bmp"):])


def binarize(image: Image):
    width, height = image.size
    [hist, _] = np.histogram(image, bins=256, range=(0, 255))
    # Normalization
    hist = 1.0 * hist / np.sum(hist)
    max_val = -999
    threshold = -1
    for t in range(1, 255):
        weight_left = np.sum(hist[:t])
        weight_right = np.sum(hist[t:])
        mean_left = np.sum(np.array([i for i in range(t)]) * hist[:t]) / weight_left
        mean_right = np.sum(np.array([i for i in range(t, 256)]) * hist[t:]) / weight_right
        val = weight_left * (1 - weight_left) * np.power(mean_left - mean_right, 2)
        if max_val < val:
            max_val = val
            threshold = t
    print(threshold)
    binarized_image = Image.new("1", (width, height), 'white')
    for y in range(height):
        for x in range(width):
            orig_pixel = image.getpixel((x, y))
            pixel = 0 if orig_pixel < threshold else 255
            binarized_image.putpixel((x, y), pixel)
    return binarized_image

def binarize_(image: Image):
    width, height = image.size
    [hist, _] = np.histogram(image, bins=256, range=(0, 255))
    max_val = float('-inf')
    threshold = -1
    for t in range(1, 255):
        weight_left = np.sum(hist[:t])
        weight_right = np.sum(hist[t:])
        mean_left = np.sum(np.array([i for i in range(t)]) * hist[:t]) / weight_left
        mean_right = np.sum(np.array([i for i in range(t, 256)]) * hist[t:]) / weight_right
        val = weight_left * weight_right * np.power(mean_left - mean_right, 2)
        if max_val < val:
            max_val = val
            threshold = t
    print(threshold)
    binarized_image = Image.new("1", (width, height), 'white')
    for y in range(height):
        for x in range(width):
            orig_pixel = image.getpixel((x, y))
            pixel = 0 if orig_pixel < threshold else 255
            binarized_image.putpixel((x, y), pixel)
    return binarized_image


#binarize(img).save(sys.argv[1][:sys.argv[1].find(".bmp")] + "_binarized" + sys.argv[1][sys.argv[1].find(".bmp"):])

print(f"Размер изображения:{img.format, img.size, img.mode, img.getpixel((0, 0))}")
