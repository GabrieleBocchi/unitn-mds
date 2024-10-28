import random
from math import sqrt

import cv2
import numpy as np
from scipy.signal import convolve2d

from attack_totallynotavirus import attacks
from defense_totallynotavirus import embedding
from detection_totallynotavirus import detection

tau = 0.71


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


def similarity(X, X_star):
    # Computes the similarity measure between the original and the new watermarks.
    norm_X = np.sqrt(np.sum(np.multiply(X, X)))
    norm_X_star = np.sqrt(np.sum(np.multiply(X_star, X_star)))

    if norm_X == 0 or norm_X_star == 0:
        return 0.0

    s = np.sum(np.multiply(X, X_star)) / (norm_X * norm_X_star)

    return s


def main():
    import glob

    images = sorted(glob.glob("./img/*.bmp"))

    for image in images:
        original_path = image
        watermarked_path = "watermarked.bmp"
        attacked_path = "attacked.bmp"

        ############### DEFENSE ###############
        image = cv2.imread("lena_grey.bmp", cv2.IMREAD_GRAYSCALE)

        watermarked = embedding("lena_grey.bmp", "totallynotavirus.npy")
        print(f"WPSNR: {wpsnr(image, watermarked)}")

        cv2.imwrite(watermarked_path, watermarked)

        ############### ATTACK ###############
        choice = random.choice([0, 1, 2, 3, 4, 5])
        # AWGN
        if choice == 0:
            print("AWGN")
            attacked = attacks(watermarked_path, "awgn", [30.0, 123])
            # cv2.imshow("attacked_awgn", attacked)
            # cv2.waitKey()
        # Gaussian blurring
        elif choice == 1:
            print("Gaussian blurring")
            attacked = attacks(watermarked_path, "blur", [(3, 2)])
            # cv2.imshow("attacked_blur", attacked)
            # cv2.waitKey()
        # Sharpening
        elif choice == 2:
            print("Sharpening")
            attacked = attacks(watermarked_path, "sharpen", [3, 0.2])
            # cv2.imshow("attacked_sharpen", attacked)
            # cv2.waitKey()
        # Median filtering
        elif choice == 3:
            print("Median filtering")
            attacked = attacks(watermarked_path, "median", [(3, 3)])
            # cv2.imshow("attacked_median", attacked)
            # cv2.waitKey()
        # Resizing
        elif choice == 4:
            print("Resizing")
            attacked = attacks(watermarked_path, "resize", [0.5])
            # cv2.imshow("attacked_resize", attacked)
            # cv2.waitKey()
        # JPEG Compression
        else:
            print("JPEG Compression")
            attacked = attacks(watermarked_path, "jpeg", [10])
            # cv2.imshow("attacked_jpeg", attacked)
            # cv2.waitKey()

        cv2.imwrite(attacked_path, attacked)

        ############### DETECTION ###############
        found, det_wpsnr = detection(original_path, watermarked_path, attacked_path)

        print(f"Found: {'yes' if found else 'no'}, wpsn: {det_wpsnr}")

        found, det_wpsnr = detection(original_path, watermarked_path, original_path)

        print(f"Found: {'yes' if found else 'no'}, wpsn: {det_wpsnr}")


if __name__ == "__main__":
    main()
