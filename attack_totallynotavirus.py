from typing import List, Union

import numpy as np


def awgn(img, std, seed):
    mean = 0.0  # some constant
    np.random.seed(seed)
    attacked = img + np.random.normal(mean, std, img.shape)
    attacked = np.clip(attacked, 0, 255)
    return attacked


def blur(img, sigma):
    from scipy.ndimage import gaussian_filter

    attacked = gaussian_filter(img, sigma)
    return attacked


def sharpening(img, sigma, alpha):
    from scipy.ndimage import gaussian_filter

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
    import random

    path = f"tmp{random.randint(0, 10)}.jpg"

    img = Image.fromarray(img)
    img = img.convert("L")
    img.save(path, "JPEG", quality=QF)
    attacked = Image.open(path)
    attacked = np.asarray(attacked, dtype=np.uint8)
    os.remove(path)
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


def find_best_attack(image_path, watermarked_path, adv_detection):
    import cv2
    import random

    random.seed(0)

    attacks_list = [
        ("awgn", [15.0, 123]),
        ("awgn", [30.0, 123]),
        ("awgn", [5.0, 123]),
        ("blur", [(3, 2)]),
        ("blur", [(2, 1)]),
        ("sharpen", [2, 0.2]),
        ("resize", [0.8]),
        ("resize", [0.5]),
        ("median", [(3, 3)]),
        ("jpeg", [30]),
        ("jpeg", [50]),
        ("jpeg", [80]),
    ]

    best_attack = None
    tmp_attack = f"./tmp{random.randint(0, 90)}.bmp"
    for i in range(len(attacks_list)):
        for j in range(len(attacks_list) + 1):
            attack = [
                attacks_list[i],
            ]
            if j < len(attacks_list):
                attack.append(attacks_list[j])

            attack_args = (
                [k[0] for k in attack],
                [k[1] for k in attack],
            )
            print(attack_args)
            attacked = attacks(watermarked_path, attack_args[0], attack_args[1])
            cv2.imwrite(tmp_attack, attacked)

            found, wpsnr = adv_detection(image_path, watermarked_path, tmp_attack)
            if not found and (best_attack is None or wpsnr > best_attack[1]):
                print(f"not found: {wpsnr}")
                best_attack = (attack_args, wpsnr)

    if best_attack is None:
        print("No attack found")
        return
    print(f"Best attack: {best_attack[0]} with WPSNR {best_attack[1]}")


if __name__ == "__main__":
    # import cv2
    from groups.ammhackati.detection_ammhackati import detection

    group_prefix = "ammhackati"
    # img_name = "rollercoaster"
    img_name = "tree"
    # img_name = "buildings"

    image_path = f"./comp_img/{img_name}.bmp"
    watermarked_path = f"./groups/{group_prefix}/{group_prefix}_{img_name}.bmp"

    find_best_attack(
        image_path,
        watermarked_path,
        detection,
    )

    # attack_args = [["median", "jpeg", "sharpen"], [[(5, 5)], [80], [15, 0.05]]]
    # attacked = attacks(
    #     watermarked_path,
    #     attack_args[0],
    #     attack_args[1],
    # )
    # cv2.imwrite("./tmp.bmp", attacked)
    #
    # found, wpsnr = detection(
    #     image_path,
    #     watermarked_path,
    #     "./tmp.bmp",
    # )
    # print(found, wpsnr)
    print(img_name)
