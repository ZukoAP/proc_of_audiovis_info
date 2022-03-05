from PIL import Image
import numpy as np
import sys

# import lab_1.main as sampling
"""
Добавить примеры с солью и перцем или с блок-схемами
"""

pattern = np.array([[[0, 0, 0],
                     [0, 1, 0],
                     [1, 1, 0]],
                    [[0, 0, 0],
                     [0, 1, 0],
                     [0, 1, 1]],
                    [[0, 0, 0],
                     [0, 1, 0],
                     [1, 1, 1]]])


def filter_image(image: Image, white: bool = False):
    width, height = image.size
    if image.mode != "1":
        return None
    filtered = Image.new("1", (width, height), 'white')
    if white:
        image_array = (np.array(image)).astype(int)
    else:
        image_array = (np.invert(np.array(image))).astype(int)
    for y in range(height):
        for x in range(width):
            print(x, y)
            pix = image.getpixel((x, y))
            if y in [0, height] or x in [0, width]:
                filtered.putpixel((x, y), pix)
                continue
            if white:
                if pix == 0:
                    filtered.putpixel((x, y), 0)
                    continue
            else:
                if pix == 255:
                    filtered.putpixel((x, y), 255)
                    continue
            box = image_array[y - 1:y + 2, x - 1:x + 2]
            if np.sum(box) in [1, 2]:
                filtered.putpixel((x, y), pix)
                continue
            if np.sum(box) > 4:
                filtered.putpixel((x, y), pix)
                continue
            for pat in pattern:
                flag = False
                for _ in range(4):
                    if np.array_equal(pat, box):
                        if white:
                            pix = 0
                        else:
                            pix = 255
                        flag = True
                        break
                    pat = np.rot90(pat)
                if flag:
                    break
            filtered.putpixel((x, y), pix)
    return filtered


def difference_image(img_1: Image, img_2: Image):
    if img_1.size != img_2.size:
        return None
    width, height = img_1.size
    difference = Image.new("1", (width, height), 'white')
    for y in range(height):
        for x in range(width):
            pix_1 = img_1.getpixel((x, y))
            pix_2 = img_2.getpixel((x, y))
            if pix_1 ^ pix_2 == 255:
                difference.putpixel((x, y), 0)
    return difference


if __name__ == "__main__":
    img = Image.open(sys.argv[1])
    # black
    '''filter_image(img).save(sys.argv[1][:sys.argv[1].find(".bmp")] + "_filtered" + sys.argv[1][sys.argv[1].find(".bmp"):])
    img_ = Image.open(sys.argv[1][:sys.argv[1].find(".bmp")] + "_filtered" + sys.argv[1][sys.argv[1].find(".bmp"):])
    difference_image(img, img_).save(sys.argv[1][:sys.argv[1].find(".bmp")] + "_difference" + sys.argv[1][sys.argv[1].find(".bmp"):])'''
    # white
    '''filter_image(img, True).save(sys.argv[1][:sys.argv[1].find(".bmp")] + "_filtered_w" + sys.argv[1][sys.argv[1].find(".bmp"):])
    img__ = Image.open(sys.argv[1][:sys.argv[1].find(".bmp")] + "_filtered_w" + sys.argv[1][sys.argv[1].find(".bmp"):])
    difference_image(img, img__).save(sys.argv[1][:sys.argv[1].find(".bmp")] + "_difference_w" + sys.argv[1][sys.argv[1].find(".bmp"):])'''
    # both
    filter_image(filter_image(img), True).save(
        sys.argv[1][:sys.argv[1].find(".bmp")] + "_fil_bw" + sys.argv[1][sys.argv[1].find(".bmp"):])
    img___ = Image.open(sys.argv[1][:sys.argv[1].find(".bmp")] + "_fil_bw" + sys.argv[1][sys.argv[1].find(".bmp"):])
    difference_image(img, img___).save(
        sys.argv[1][:sys.argv[1].find(".bmp")] + "_diff_bw" + sys.argv[1][sys.argv[1].find(".bmp"):])
