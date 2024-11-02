import glob
import random

import cv2
import numpy as np
import pywt
from matplotlib import pyplot as plt
from sklearn.metrics import auc, roc_curve

from defense_totallynotavirus import embedding
from attack_totallynotavirus import attacks


# def similarity(X, X_star):
#     return len([x for x, y in zip(X, X_star) if x == y]) / len(X)
def similarity(X, X_star):
    # Computes the similarity measure between the original and the new watermarks.
    norm_X = np.sqrt(np.sum(np.multiply(X, X)))
    norm_X_star = np.sqrt(np.sum(np.multiply(X_star, X_star)))

    if norm_X == 0 or norm_X_star == 0:
        return 0.0

    s = np.sum(np.multiply(X, X_star)) / (norm_X * norm_X_star)

    return s


# ad-hoc detection function to deal with ROC calculation
def detection(image, watermarked_image, mark_size):
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

    # Define the subbands to process
    subbands = [
        ((LH_ori, LH_wat), "1"),
        ((HL_ori, HL_wat), "1"),
        ((HH_ori, HH_wat), "1"),
        ((LL_LH_ori, LL_LH_wat), "2"),
        ((LL_HL_ori, LL_HL_wat), "2"),
        ((LL_HH_ori, LL_HH_wat), "2"),
        ((LL_LL_LH_ori, LL_LL_LH_wat), "3"),
        ((LL_LL_HL_ori, LL_LL_HL_wat), "3"),
        ((LL_LL_HH_ori, LL_LL_HH_wat), "3"),
        ((LL_LL_LL_ori, LL_LL_LL_wat), "4"),
    ]

    extracted_marks = []

    ALPHAS = {
        "1": 0.003,
        "2": 0.002,
        "3": 0.001,
        "4": 0.0005,
    }

    # Process each subband
    for (band_ori, band_wat), depth in subbands:
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

        extracted_mark = np.zeros(mark_size, dtype=np.float64)
        touched = [0 for _ in range(mark_size)]

        if len(locations) < mark_size:
            continue

        for i, loc in enumerate(locations):
            # extracted_mark[i % mark_size] += (
            #     band_wat[loc] / band_ori[loc] - 1
            # ) / ALPHAS[depth]
            extracted_mark[i % mark_size] += (band_wat[loc] - band_ori[loc]) / ALPHAS[
                depth
            ]
            touched[i % mark_size] += 1

        extracted_mark /= touched
        extracted_mark = np.where(extracted_mark > 0.5, 1, 0)
        extracted_marks.append(extracted_mark)

    extracted_mark = np.clip(np.mean(extracted_marks, axis=0), 0, 1)

    return extracted_mark


# some parameters for the spread spectrum
mark_size = 1024
np.random.seed(seed=124)
random.seed(123)


# scores and labels are two lists we will use to append the values of similarity and their labels
# In scores we will append the similarity between our watermarked image and the attacked one,
# or  between the attacked watermark and a random watermark
# In labels we will append the 1 if the scores was computed between the watermarked image and the attacked one,
# and 0 otherwise
scores = []
labels = []
index = 0

images = sorted(glob.glob("./img/*.bmp"))
mark = np.load("totallynotavirus.npy")

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

watermarked_path = "watermarked.bmp"

for i, img_path in enumerate(images):
    print(f"Processing {i}/{len(images)}")
    image = cv2.imread(img_path, 0)

    # Embed Watermark
    watermarked = embedding(img_path, "totallynotavirus.npy")
    cv2.imwrite(watermarked_path, watermarked)

    index += 1

    sample = 0
    while sample < 10:
        # fakemark is the watermark for H0
        fakemark = np.random.uniform(0.0, 1.0, mark_size)
        fakemark = np.uint8(np.rint(fakemark))
        # random attack to watermarked image
        random_attack = random.choice(attacks_list)
        res_att = attacks(watermarked_path, random_attack[0], random_attack[1])
        # extract attacked watermark

        wat_attacked = detection(image, res_att, mark_size)
        wat_extracted = detection(image, watermarked, mark_size)
        # compute similarity H1
        scores.append(similarity(wat_extracted, wat_attacked))
        labels.append(1)
        # compute similarity H0
        scores.append(similarity(fakemark, wat_attacked))
        labels.append(0)
        sample += 1


# compute ROC
fpr, tpr, tau = roc_curve(
    np.asarray(labels), np.asarray(scores), drop_intermediate=False
)
# compute AUC
roc_auc = auc(fpr, tpr)
plt.figure()
lw = 2

plt.plot(fpr, tpr, color="darkorange", lw=lw, label="AUC = %0.2f" % roc_auc)
plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
plt.xlim([-0.01, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver operating characteristic example")
plt.legend(loc="lower right")
plt.show()
idx_tpr = np.where((fpr - 0.05) == min(i for i in (fpr - 0.05) if i > 0))
print(
    "For a FPR approximately equals to 0.05 corresponds a TPR equals to %0.2f"
    % tpr[idx_tpr[0][0]]
)
print(
    "For a FPR approximately equals to 0.05 corresponds a threshold equals to %0.2f"
    % tau[idx_tpr[0][0]]
)
print("Check FPR %0.2f" % fpr[idx_tpr[0][0]])
