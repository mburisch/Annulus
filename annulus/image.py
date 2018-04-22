
import cv2
import numpy as np

def binarize(image, block_size = 33):
    """Binarizes an image using OpenCV adaptive threshold.

    Args:
        image:      Grayscale image to binarize
        block_size: Block size for thresholding

    Returns:
        Binarized image
    """

    return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, 0)


def threshold_image(image, block_size = (64, 64), step_size = (32, 32), min_range = 32, min_value= 64):
    """Binarizes an image using Otsu thresholding.
   
    For each block_size check if there is a sufficient gray value range (min_range). If it is the case
    use Otsu thresholding to binarize. If not check if the gray values are above a certain minimum
    (min_value) threshold. This avoids binarization problems in homogeneous regions.

    Args:
        image:      Grayscale image to binarize
        block_size: Block size for thresholding
        step_size:  Step size for blocks
        min_range:  Minimum range between minimum and maximum gray value for use of Otsu.
        min_value:  Minimum value in block to consider everything as white.

    Returns:
        Binarized image
    """

    binary = np.zeros_like(image)
    for row in range(0, image.shape[0], step_size[0]):
        for col in range(0, image.shape[1], step_size[1]):
            im = image[row:row + block_size[0], col:col + block_size[1]]
            if np.ptp(im) < min_range:
                if np.min(im) > min_value:
                    binary[row:row + block_size[0], col:col + block_size[1]] = 255
            else:
                _, bin = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                binary[row:row + block_size[0], col:col + block_size[1]] = bin
    return binary