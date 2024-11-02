import glob
import multiprocessing
from math import sqrt
from pathlib import Path

import cv2
import numpy as np
from scipy.signal import convolve2d

from attack_totallynotavirus import attacks
from defense_totallynotavirus import embedding
from detection_totallynotavirus import detection


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


def test_image(schedule_sem, img_path, i, attacks_list, q):
    schedule_sem.acquire()

    watermarked_path = f"./tmp/watermarked{i}.bmp"
    false_watermarked_path = f"./tmp/false_watermarked{i}.bmp"
    attacked_path = f"./tmp/attacked{i}.bmp"

    ############### DEFENSE ###############
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    watermarked = embedding(img_path, "totallynotavirus.npy")

    result = {}
    result["i"] = i
    result["image"] = img_path

    result["wpsnr"] = wpsnr(image, watermarked)

    cv2.imwrite(watermarked_path, watermarked)

    ############### DETECTION OF WATERMARKED IMAGE ###############
    result["watermarked"], det_wpsnr = detection(
        img_path, watermarked_path, watermarked_path
    )

    ############### DETECTION OF FALSE POSITIVES ###############
    result["not_watermarked"], det_wpsnr = detection(
        img_path, watermarked_path, img_path
    )

    mark = np.random.uniform(low=0.0, high=1.0, size=1024)
    mark = np.uint8(np.rint(mark))
    np.save("mark.npy", mark)
    false_watermarked = embedding(img_path, "mark.npy")
    cv2.imwrite(false_watermarked_path, false_watermarked)

    result["fake_watermarked"], det_wpsnr = detection(
        img_path, watermarked_path, false_watermarked_path
    )

    result["failing_wpsnr"] = []
    result["attack_types"] = {}
    result["points"] = []
    result["under_25"] = 0

    ############### ATTACK ###############
    for attack, params in attacks_list:
        attacked = attacks(watermarked_path, attack, params)

        cv2.imwrite(attacked_path, attacked)

        ############### DETECTION ###############
        found, det_wpsnr = detection(img_path, watermarked_path, attacked_path)

        if found and det_wpsnr <= 25:
            result["under_25"] += 1
        if not found and det_wpsnr > 35:
            result["failing_wpsnr"].append(det_wpsnr)
            if attack not in result["attack_types"]:
                result["attack_types"][attack] = []
            result["attack_types"][attack].append(det_wpsnr)

        if det_wpsnr < 38:
            result["points"].append(6)
        elif det_wpsnr < 41:
            result["points"].append(5)
        elif det_wpsnr < 44:
            result["points"].append(4)
        elif det_wpsnr < 47:
            result["points"].append(3)
        elif det_wpsnr < 50:
            result["points"].append(2)
        elif det_wpsnr < 53:
            result["points"].append(1)

    schedule_sem.release()
    q.put(result)


def main():
    tmp = Path("./tmp")
    if not tmp.exists():
        tmp.mkdir()

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

    with multiprocessing.Manager() as manager:
        q = manager.Queue()
        schedule_sem = multiprocessing.Semaphore(6)
        processes = [
            multiprocessing.Process(
                target=test_image,
                args=(
                    schedule_sem,
                    image,
                    i,
                    attacks_list,
                    q,
                ),
            )
            for i, image in enumerate(images)
        ]

        for process in processes:
            process.start()

        wpsnr_list = []
        severe_watermark_not_detected_list = []
        not_watermarked_false_positives_list = []
        severe_fake_watermarked_list = []
        failing_wpsnr_list = []
        points_list = []
        under_25 = 0

        for _ in processes:
            proc_res = q.get()
            print(f"Thread {proc_res['i']} done")
            wpsnr_list.append(proc_res["wpsnr"])
            if not proc_res["watermarked"]:
                severe_watermark_not_detected_list.append(proc_res["image"])
            if proc_res["not_watermarked"]:
                not_watermarked_false_positives_list.append(proc_res["image"])
            if proc_res["fake_watermarked"]:
                severe_fake_watermarked_list.append(proc_res["image"])
            failing_wpsnr_list.extend(proc_res["failing_wpsnr"])
            points_list.extend(proc_res["points"])
            for attack, values in proc_res["attack_types"].items():
                attack_types[attack].extend(values)

            under_25 += proc_res["under_25"]

    print(f"Average WPSNR after watermarking: {sum(wpsnr_list) / len(images)}")
    print(
        f"Average on failing tests WPSNR: {sum(failing_wpsnr_list) / max(len(failing_wpsnr_list), 1)}"
    )
    print(f"Points: {sum(points_list) / max(len(points_list), 1)}")
    print(f"Detected watermark under 25 wpsnr: {under_25}")
    print(
        f"Watermark not detected in same image {len(severe_watermark_not_detected_list)} images: {severe_watermark_not_detected_list}"
    )
    print(
        f"False positives in non watermarked images: {len(not_watermarked_false_positives_list)}: {not_watermarked_false_positives_list}"
    )
    print(
        f"False positives in false watermarked images: {len(severe_fake_watermarked_list)}: {severe_fake_watermarked_list}"
    )
    # print(f"False negatives: {len(failing_wpsnr_list)}: {failing_wpsnr_list}")

    for attack, values in attack_types.items():
        if values:
            print(f"Attack {attack} WPSNR: {sum(values) / len(values)}")

    for f in tmp.glob("*"):
        f.unlink()
    tmp.rmdir()


if __name__ == "__main__":
    main()
