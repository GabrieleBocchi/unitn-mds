from typing import List, Union

import numpy as np


def awgn(img, std, seed):
    mean = 0.0  # some constant
    np.random.seed(seed)
    attacked = img + np.random.normal(mean, std, img.shape)
    attacked = np.clip(attacked, 0, 255)
    return attacked


def blur(img, sigma):
    from scipy.ndimage.filters import gaussian_filter

    attacked = gaussian_filter(img, sigma)
    return attacked


def sharpening(img, sigma, alpha):
    from scipy.ndimage import gaussian_filter

    # print(img/255)
    filter_blurred_f = gaussian_filter(img, sigma)

    attacked = img + alpha * (img - filter_blurred_f)
    return attacked


def median(img, kernel_size):
    from scipy.signal import medfilt

    attacked = medfilt(img, kernel_size)
    return attacked


def resizing(img, scale):
    from skimage.transform import rescale

    x, y = img.shape
    attacked = rescale(img, scale)
    attacked = rescale(attacked, 1 / scale)
    attacked = attacked[:x, :y]
    return attacked


def jpeg_compression(img, QF):
    import os

    from PIL import Image

    img = Image.fromarray(img)
    img.save("tmp.jpg", "JPEG", quality=QF)
    attacked = Image.open("tmp.jpg")
    attacked = np.asarray(attacked, dtype=np.uint8)
    os.remove("tmp.jpg")
    return attacked


def attacks(input1: str, attack_name: Union[str, List[str]], param_array: List):
    """
    Perform some processing techniques in order to (try to) extract the watermark from the image

    :param input1: filename of the watermarkED image
    :param attack_name: string name of the attack to pergorm. Can be a list
    :param param_array: array of specific parameters
    :return: the attacked image
    """

    import cv2

    image = cv2.imread(input1, cv2.IMREAD_GRAYSCALE)

    if isinstance(attack_name, str):
        attack_name = [attack_name]
        param_array = [param_array]

    for name, param in zip(attack_name, param_array):
        if name == "awgn":
            image = awgn(image, param[0], param[1])
        elif name == "blur":
            image = blur(image, param[0])
        elif name == "sharpen":
            image = sharpening(image, param[0], param[1])
        elif name == "median":
            image = median(image, param[0])
        elif name == "resize":
            image = resizing(image, param[0])
        elif name == "jpeg":
            image = jpeg_compression(image, param[0])
        else:
            raise ValueError(f"Attack {name} not recognized")

    return image
