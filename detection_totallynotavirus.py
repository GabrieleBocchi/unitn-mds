from math import sqrt

import cv2
import numpy as np
import pywt
from scipy.signal import convolve2d


def wpsnr(img1, img2):
    img1 = np.float32(img1) / 255.0
    img2 = np.float32(img2) / 255.0
    difference = img1 - img2
    same = not np.any(difference)
    if same is True:
        return 9999999
    w = np.genfromtxt("csf.csv", delimiter=",")
    ew = convolve2d(difference, np.rot90(w, 2), mode="valid")
    decibels = 20.0 * np.log10(1.0 / sqrt(np.mean(np.mean(ew**2))))
    return decibels


def detection(input1, input2, input3):
    """
    Check if the watermark is still present or not in the attached image.

    :param input1: Path to the original image.
    :param input2: Path to the watermarked image.
    :param input3: Path to the attacked image.
    :return: Watermark detected (1 or 0) and WPSNR.
    """
    mark_size = 1024

    # Load all inputs
    image = cv2.imread(input1, cv2.IMREAD_GRAYSCALE)
    watermarked_image = cv2.imread(input2, cv2.IMREAD_GRAYSCALE)
    attacked_image = cv2.imread(input3, cv2.IMREAD_GRAYSCALE)

    # Perform 2D DWT on the original image
    LL_ori, (LH_ori, HL_ori, HH_ori) = pywt.dwt2(image, "haar")
    LL_LL_ori, (LL_LH_ori, LL_HL_ori, LL_HH_ori) = pywt.dwt2(LL_ori, "haar")
    LL_LL_LL_ori, (LL_LL_LH_ori, LL_LL_HL_ori, LL_LL_HH_ori) = pywt.dwt2(
        LL_LL_ori, "haar"
    )

    # Perform 2D DWT on the watermarked image
    LL_wat, (LH_wat, HL_wat, HH_wat) = pywt.dwt2(watermarked_image, "haar")
    LL_LL_wat, (LL_LH_wat, LL_HL_wat, LL_HH_wat) = pywt.dwt2(LL_wat, "haar")
    LL_LL_LL_wat, (LL_LL_LH_wat, LL_LL_HL_wat, LL_LL_HH_wat) = pywt.dwt2(
        LL_LL_wat, "haar"
    )

    # Perform 2D DWT on the attacked image
    LL_att, (LH_att, HL_att, HH_att) = pywt.dwt2(attacked_image, "haar")
    LL_att, (LL_LH_att, LL_HL_att, LL_HH_att) = pywt.dwt2(LL_att, "haar")
    LL_LL_LL_att, (LL_LL_LH_att, LL_LL_HL_att, LL_LL_HH_att) = pywt.dwt2(LL_att, "haar")

    # Define the subbands to process
    subbands = [
        ((LH_ori, LH_wat, LH_att), "1"),
        ((HL_ori, HL_wat, HL_att), "1"),
        ((HH_ori, HH_wat, HH_att), "1"),
        ((LL_LH_ori, LL_LH_wat, LL_LH_att), "2"),
        ((LL_HL_ori, LL_HL_wat, LL_HL_att), "2"),
        ((LL_HH_ori, LL_HH_wat, LL_HH_att), "2"),
        ((LL_LL_LH_ori, LL_LL_LH_wat, LL_LL_LH_att), "3"),
        ((LL_LL_HL_ori, LL_LL_HL_wat, LL_LL_HL_att), "3"),
        ((LL_LL_HH_ori, LL_LL_HH_wat, LL_LL_HH_att), "3"),
        ((LL_LL_LL_ori, LL_LL_LL_wat, LL_LL_LL_att), "4"),
    ]

    extracted_marks = []
    attacked_marks = []

    ALPHAS = {
        "1": 0.003,
        "2": 0.002,
        "3": 0.001,
        "4": 0.0005,
    }

    # Process each subband
    for (band_ori, band_wat, band_att), depth in subbands:
        abs_band_ori = abs(band_ori).flatten()

        position_value = [
            (i, abs_band_ori[i])
            for i in range(len(abs_band_ori))
            if abs_band_ori[i] > 1
        ]
        position_value.sort(key=lambda x: x[1], reverse=True)
        locations = [
            (
                position_value[i][0] // band_ori.shape[0],
                position_value[i][0] % band_ori.shape[1],
            )
            for i in range(len(position_value))
        ]

        if len(locations) < mark_size:
            continue

        extracted_mark = np.zeros(mark_size, dtype=np.float64)
        attacked_mark = np.zeros(mark_size, dtype=np.float64)
        touched = [0 for _ in range(mark_size)]

        for i, loc in enumerate(locations):
            extracted_mark[i % mark_size] += (band_wat[loc] - band_ori[loc]) / ALPHAS[
                depth
            ]
            attacked_mark[i % mark_size] += (band_att[loc] - band_ori[loc]) / ALPHAS[
                depth
            ]
            touched[i % mark_size] += 1

        extracted_mark /= touched
        attacked_mark /= touched
        extracted_mark = np.where(extracted_mark > 0.5, 1, 0)
        extracted_marks.append(extracted_mark)
        attacked_mark = np.where(attacked_mark > 0.5, 1, 0)
        attacked_marks.append(attacked_mark)

    extracted_mark = np.where(np.mean(extracted_marks, axis=0) > 0.75, 1, 0)
    attacked_mark = np.clip(np.mean(attacked_marks, axis=0), 0, 1)

    # if similarity is greater then the threshold return 1, wpsnr of the attacked image
    return similarity(extracted_mark, attacked_mark) > 0.7, wpsnr(
        attacked_image, watermarked_image
    )


def similarity(X, X_star):
    # Computes the similarity measure between the original and the new watermarks.
    norm_X = np.sqrt(np.sum(np.multiply(X, X)))
    norm_X_star = np.sqrt(np.sum(np.multiply(X_star, X_star)))

    if norm_X == 0 or norm_X_star == 0:
        return 0.0

    s = np.sum(np.multiply(X, X_star)) / (norm_X * norm_X_star)

    return s
