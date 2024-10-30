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

    failing_wpsnr = []
    after_water_wpsnr = 0
    points = []
    watermark_not_detected = []
    false_positives_image_non_watermarked = []
    false_positives_image_false_watermarked = []

    images = sorted(glob.glob("./img/*.bmp"))

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
        ("jpeg", [50]),
        ("jpeg", [80]),
    ]

    attack_types = {}

    for attack_type, _ in attacks_list:
        attack_types[attack_type] = []

    for i, image in enumerate(images):
        print(f"Image {i + 1}/{len(images)}")
        original_path = image
        watermarked_path = "watermarked.bmp"
        false_watermarked_path = "false_watermarked.bmp"
        attacked_path = "attacked.bmp"

        ############### DEFENSE ###############
        image = cv2.imread(original_path, cv2.IMREAD_GRAYSCALE)

        watermarked = embedding(original_path, "totallynotavirus.npy")

        after_water_wpsnr += wpsnr(image, watermarked)

        cv2.imwrite(watermarked_path, watermarked)

        ############### DETECTION OF WATERMARKED IMAGE ###############
        print("Testing watermarked image")
        found, det_wpsnr = detection(original_path, watermarked_path, watermarked_path)
        if not found:
            watermark_not_detected.append(original_path.split("/")[-1])
            print("WATERMARK NOT DETECTED")

        ############### DETECTION OF FALSE POSITIVES ###############
        print("Testing non watermarked image")
        found, det_wpsnr = detection(original_path, watermarked_path, original_path)
        if found:
            false_positives_image_non_watermarked.append(original_path.split("/")[-1])
            print("FALSE POSITIVE DETECTED IN NON WATERMARKED IMAGE")

        mark = np.random.uniform(low=0.0, high=1.0, size=1024)
        mark = np.uint8(np.rint(mark))
        np.save("mark.npy", mark)
        false_watermarked = embedding(original_path, "mark.npy")
        cv2.imwrite(false_watermarked_path, false_watermarked)

        print("Testing false watermarked image")
        found, det_wpsnr = detection(
            original_path, watermarked_path, false_watermarked_path
        )

        if found:
            false_positives_image_false_watermarked.append(original_path.split("/")[-1])
            print("FALSE POSITIVE DETECTED IN FALSE WATERMARKED IMAGE")

        print("Attacking")
        ############### ATTACK ###############
        for attack, params in attacks_list:
            attacked = attacks(watermarked_path, attack, params)

            cv2.imwrite(attacked_path, attacked)

            ############### DETECTION ###############
            found, det_wpsnr = detection(original_path, watermarked_path, attacked_path)

            if found and det_wpsnr <= 25:
                print("We shouldnt find this")
            elif not found and det_wpsnr > 35:
                print(f"FALSE NEGATIVE DETECTED, wpsnr {det_wpsnr}")
                failing_wpsnr.append(det_wpsnr)

                attack_types[attack].append(det_wpsnr)

            if det_wpsnr < 38:
                points.append(6)
            elif det_wpsnr < 41:
                points.append(5)
            elif det_wpsnr < 44:
                points.append(4)
            elif det_wpsnr < 47:
                points.append(3)
            elif det_wpsnr < 50:
                points.append(2)
            elif det_wpsnr < 53:
                points.append(1)

    print(f"Average WPSNR after watermarking: {after_water_wpsnr / len(images)}")
    print(
        f"Average on failing tests WPSNR: {sum(failing_wpsnr) / max(len(failing_wpsnr), 1)}"
    )
    print(f"Points: {sum(points) / max(len(points), 1)}")
    print(
        f"Watermark not detected in {len(watermark_not_detected)} images: {watermark_not_detected}"
    )
    print(
        f"False positives in non watermarked images: {len(false_positives_image_non_watermarked)}: {false_positives_image_non_watermarked}"
    )
    print(
        f"False positives in false watermarked images: {len(false_positives_image_false_watermarked)}: {false_positives_image_false_watermarked}"
    )
    print(f"False negatives: {len(failing_wpsnr)}: {failing_wpsnr}")

    for attack, values in attack_types.items():
        if values:
            print(f"Attack {attack} WPSNR: {sum(values) / len(values)}")


if __name__ == "__main__":
    main()
