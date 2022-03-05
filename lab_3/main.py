import numpy
from PIL import Image
import numpy as np
import sys
import lab_1.main as sampling

img = Image.open(sys.argv[1])

# define kernal convolution function
# Prewitt Operator Mask
Gx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
Gy = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])

operator_spread = max([np.sum(x) for x in Gy])


# image X and filter F
def convolve(image_array, filter):
    """
    Convolution function a way of multiplying together two arrays of numbers, generally of different sizes,
    here grayscale image and sobel/prewitt kernels, to produce a third array of numbers of the same dimensionality.
    The convolution is performed by sliding the applied kernel over all the pixels of the image.

    Parameters
    ----------
    image_array : numpy.ndarray
        numpy ndarray of grayscale image
    filter: ndarray
        kernal of convolution function

    Returns
    -------
    result : numpy.ndarray
        ndarray of image_array with applied filter
    """

    # height and width of the image
    height = image_array.shape[0]
    width = image_array.shape[1]

    # height and width of the filter
    filter_height = filter.shape[0]
    filter_width = filter.shape[1]

    # filter's distances from centre to edges, i.e. indents from image edges
    vertical_indent = (filter_height - 1) // 2
    horizontal_indent = (filter_width - 1) // 2

    # output numpy matrix with height and width
    result = np.zeros((height, width))
    # iterate over all the pixel of image
    for i in np.arange(vertical_indent, height - vertical_indent):
        for j in np.arange(horizontal_indent, width - horizontal_indent):
            total = 0
            # iterate over the filter
            for k in np.arange(-vertical_indent, vertical_indent + 1):
                for l in np.arange(-horizontal_indent, horizontal_indent + 1):
                    # get the corresponding value from image and filter
                    window = image_array[i + k, j + l]
                    filtered = filter[vertical_indent + k, horizontal_indent + l]
                    total += (window * filtered)
            result[i, j] = total
    return result


def detect_edges(img: Image, gx, gy, invert_threshold: bool = False):
    print(img.mode)
    if img.mode in ["RGB", "RGBA"]:
        img = sampling.grayscale(img, True)
        img.save(sys.argv[1][:sys.argv[1].find(".bmp")] + "_gray" + sys.argv[1][sys.argv[1].find(".bmp"):])
    elif img.mode != "L":
        return None

    image_array = (np.array(img)).astype(int)

    # normalizing the vectors and mapping to 0 .. 255
    gx_norm = convolve(image_array, Gx) / (2 * operator_spread)
    gx_map = (gx_norm + np.abs(np.min(gx_norm)))
    gx_map = (gx_map / np.max(gx_map)) * 255
    Image.fromarray(gx_map.astype('uint8'), 'L').save(sys.argv[1][:sys.argv[1].find(".bmp")] + "_edges_gx_norm" + sys.argv[1][sys.argv[1].find(".bmp"):])

    gy_norm = convolve(image_array, Gy) / (2 * operator_spread)
    gy_map = (gy_norm + np.abs(np.min(gy_norm)))
    gy_map = (gy_map / np.max(gy_map)) * 255
    Image.fromarray(gy_map.astype('uint8'), 'L').save(sys.argv[1][:sys.argv[1].find(".bmp")] + "_edges_gy_norm" + sys.argv[1][sys.argv[1].find(".bmp"):])

    gradient = np.abs(gx_norm) + np.abs(gy_norm)
    gradient_norm = (gradient / np.max(gradient)) * 255
    img_grad_norm = Image.fromarray(gradient_norm.astype('uint8'), 'L')
    img_grad_norm.save(sys.argv[1][:sys.argv[1].find(".bmp")] + "_edges_g_norm" + sys.argv[1][sys.argv[1].find(".bmp"):])

    return sampling.binarize(img_grad_norm, invert_threshold)


if __name__ == "__main__":
    detect_edges(img, Gx, Gy).save(sys.argv[1][:sys.argv[1].find(".bmp")] + "_edges_g_bin" + sys.argv[1][sys.argv[1].find(".bmp"):])
    # detect_edges(img, Gx, Gy, True).save(sys.argv[1][:sys.argv[1].find(".bmp")] + "_edges_g_bin_inv" + sys.argv[1][sys.argv[1].find(".bmp"):])
