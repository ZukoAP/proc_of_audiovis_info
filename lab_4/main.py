from PIL import Image, ImageDraw, ImageFont
from lab_1.main import binarize

W, H = (100, 100)


def generate_letters(font_path, font_size: int):
    font = ImageFont.truetype(font_path, font_size)
    for i in range(ord('a'), ord('z') + 1):
        w, h = font.getsize(chr(i))
        img = Image.new('L', (w, h), 'white')
        draw = ImageDraw.Draw(img)
        w, h = draw.textsize(chr(i), font=font)
        draw.text((w / 2, h / 2), chr(i), font=font, fill='black', anchor="mm")
        crop_letter(binarize(img)).save(f'./letters/{chr(i)}.bmp')  # average otsu threshold = 137


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


def get_features(img: Image):
    w, h = img.size
    area = w*h
    weight_b = weight_black(img)
    weight_b_rel = weight_b / area
    weight_w = area - weight_b

def gravity_centre(img:Image):
    w, h = img.size
    centre_x = 0
    centre_y = 0
    for y in range(h):
        for x in range(w):
            if img.getpixel((x, y)) == 0:
                centre_x += x


#generate_letters('C:/Windows/Fonts/timesi.ttf', 52)
