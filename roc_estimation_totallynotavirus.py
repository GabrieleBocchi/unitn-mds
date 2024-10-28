import glob
import os
import random

import cv2
import numpy as np
import pywt
from defense_totallynotavirus import embedding
from matplotlib import pyplot as plt
from numpy.linalg import norm
from PIL import Image
from scipy.linalg import svd
from scipy.ndimage import gaussian_filter
from scipy.signal import medfilt
from scipy.spatial.distance import cosine
from skimage.metrics import mean_squared_error
from skimage.metrics import structural_similarity as ssim
from skimage.transform import rescale
from sklearn.metrics import auc, roc_curve
from wpsnr import wpsnr


# Function to compute the SVD
def compute_svd(matrix):
    U, S, Vt = svd(matrix, full_matrices=False)
    return U, S, Vt


# Function to compare singular values using Frobenius norm
def compare_singular_values(S1, S2):
    return norm(S1 - S2)


# Function to reconstruct the watermark from its SVD components
def reconstruct_from_svd(U, S, Vt):
    return np.dot(U, np.dot(np.diag(S), Vt))


# Function to compare original and candidate watermarks
def compare_watermarks(original, candidate):
    mse_value = mean_squared_error(original, candidate)
    original_resized = cv2.resize(original, (7, 7), interpolation=cv2.INTER_LINEAR)
    candidate_resized = cv2.resize(candidate, (7, 7), interpolation=cv2.INTER_LINEAR)
    ssim_value = ssim(original_resized, candidate_resized, data_range=1)
    return mse_value, ssim_value


def singular_value_similarity(A, B):
    # Compute the SVD
    _, s_B, _ = np.linalg.svd(B, full_matrices=False)

    # Compare the singular values (you can use other distance metrics)
    return np.linalg.norm(A[1] - s_B)


def vector_similarity(A, B):
    # Compute the SVD
    U_A, _, V_A = A
    U_B, _, V_B = B

    # Compute cosine similarity between U and V (or pick one of the matrices)
    u_similarity = np.mean(
        [1 - cosine(U_A[:, i], U_B[:, i]) for i in range(U_A.shape[1])]
    )
    v_similarity = np.mean(
        [1 - cosine(V_A[:, i], V_B[:, i]) for i in range(V_A.shape[1])]
    )

    return (u_similarity + v_similarity) / 2


def awgn(img, std, _):
    mean = 0.0  # some constant
    # np.random.seed(seed)
    attacked = img + np.random.normal(mean, std, img.shape)
    attacked = np.clip(attacked, 0, 255)
    return attacked


def blur(img, sigma):
    attacked = gaussian_filter(img, sigma)
    return attacked


def sharpening(img, sigma, alpha):
    filter_blurred_f = gaussian_filter(img, sigma)

    attacked = img + alpha * (img - filter_blurred_f)
    return attacked


def median(img, kernel_size):

    attacked = medfilt(img, kernel_size)
    return attacked


def resizing(img, scale):

    x, y = img.shape
    attacked = rescale(img, scale)
    attacked = rescale(attacked, 1 / scale)
    attacked = attacked[:x, :y]
    return attacked


def jpeg_compression(img, QF):

    img = Image.fromarray(img)
    img.save("tmp.jpg", "JPEG", quality=QF)
    attacked = Image.open("tmp.jpg")
    attacked = np.asarray(attacked, dtype=np.uint8)
    os.remove("tmp.jpg")

    return attacked


def random_attack(img):
    i = random.randint(1, 7)
    if i == 1:
        attacked = awgn(img, 3.0, 123)
    elif i == 2:
        attacked = blur(img, [3, 3])
    elif i == 3:
        attacked = sharpening(img, 1, 1)
    elif i == 4:
        attacked = median(img, [3, 3])
    elif i == 5:
        attacked = resizing(img, 0.8)
    elif i == 6:
        attacked = jpeg_compression(img, 75)
    elif i == 7:
        attacked = img
    else:
        raise ValueError("Invalid attack index")
    return attacked


def similarity(X, X_star):
    # Computes the similarity measure between the original and the new watermarks.
    norm_X = np.sqrt(np.sum(np.multiply(X, X)))
    norm_X_star = np.sqrt(np.sum(np.multiply(X_star, X_star)))

    if norm_X == 0 or norm_X_star == 0:
        return 0.0

    s = np.sum(np.multiply(X, X_star)) / (norm_X * norm_X_star)

    return s


# ad-hoc detection function to deal with ROC calculation
def detection(image, watermarked_image, alpha, mark_size, v, watermark_svd):
    # Perform 2D DWT on the watermarked image
    _, (LH1, _, _) = pywt.dwt2(watermarked_image, "haar")
    LL2, (_, _, _) = pywt.dwt2(LH1, "haar")
    _, (LH3, HL3, HH3) = pywt.dwt2(LL2, "haar")

    _, (LH1_image, _, _) = pywt.dwt2(image, "haar")
    LL2_image, _ = pywt.dwt2(LH1_image, "haar")
    _, (LH3_image, HL3_image, HH3_image) = pywt.dwt2(LL2_image, "haar")

    # Define the subbands to process
    subbands = {"LH": (LH3, LH3_image), "HL": (HL3, HL3_image), "HH": (HH3, HH3_image)}
    extracted_marks = []

    # Process each subband
    for _, (band, band_image) in subbands.items():
        extracted_mark = np.zeros(mark_size)
        # Get the locations in the subband
        abs_band = abs(band)
        locations_band = np.argsort(-abs_band, axis=None)  # Descending order
        rows_band = band.shape[0]
        locations_band = [
            (val // rows_band, val % rows_band) for val in locations_band
        ]  # (x, y) coordinates

        # Extract the watermark from the sub-band
        for idx, loc in enumerate(
            locations_band[1 : mark_size + 1]
        ):  # Skip the first location
            if idx >= mark_size:
                print(f"IDX {idx} bigger than mark size {mark_size}")
                break
            if v == "additive":
                extracted_mark[idx] = (abs_band[loc] - band_image[loc]) / (alpha)
            elif v == "multiplicative":
                extracted_mark[idx] = (abs_band[loc] - band_image[loc]) / (
                    alpha * band_image[loc]
                )

        extracted_mark = np.clip(extracted_mark, 0, 1)
        extracted_marks.append(extracted_mark)

    singular_value_diffs = []
    mse_ssim_scores = []

    best_candidate = None
    highest_sim = -1

    for candidate in extracted_marks:
        candidate_matrix = np.reshape(candidate, (mark_size, 1))
        # Compute SVD for the candidate watermark
        U_cand, S_cand, Vt_cand = compute_svd(candidate_matrix)

        # 1. Singular value comparison
        sv_diff = compare_singular_values(watermark_svd[1], S_cand)
        singular_value_diffs.append(sv_diff)

        # 2. Reconstruct watermarks
        original_reconstructed = reconstruct_from_svd(
            watermark_svd[0], watermark_svd[1], watermark_svd[2]
        )
        candidate_reconstructed = reconstruct_from_svd(U_cand, S_cand, Vt_cand)

        # 3. Compare reconstructed watermarks (MSE and SSIM)
        mse_val, ssim_val = compare_watermarks(
            original_reconstructed, candidate_reconstructed
        )

        sim3 = vector_similarity(watermark_svd, (U_cand, S_cand, Vt_cand))
        mse_ssim_scores.append((mse_val, ssim_val))

        if sim3 > highest_sim:
            highest_sim = sim3
            best_candidate = candidate

    return best_candidate


# some parameters for the spread spectrum
mark_size = 1024
alpha = 0.3
v = "additive"
np.random.seed(seed=124)


# scores and labels are two lists we will use to append the values of similarity and their labels
# In scores we will append the similarity between our watermarked image and the attacked one,
# or  between the attacked watermark and a random watermark
# In labels we will append the 1 if the scores was computed between the watermarked image and the attacked one,
# and 0 otherwise
scores = []
labels = []
index = 0

images = sorted(glob.glob("./img/*.bmp"))

for img_path in images:
    image = cv2.imread(img_path, 0)

    # Embed Watermark
    watermarked = embedding(img_path, "totallynotavirus.npy")
    mark = np.load("totallynotavirus.npy")
    # print(f"Watermarked {index} WSPNR: {wpsnr(image, watermarked)}")

    index += 1
    watermark_matrix = mark.reshape((mark_size, 1))
    watermark_svd = compute_svd(watermark_matrix)

    sample = 0
    while sample < 1:
        # fakemark is the watermark for H0
        fakemark = np.random.uniform(0.0, 1.0, mark_size)
        fakemark = np.uint8(np.rint(fakemark))
        # random attack to watermarked image
        res_att = random_attack(watermarked)
        # print(f"Attacked WSPNR: {wpsnr(res_att, watermarked)}")
        # extract attacked watermark
        wat_attacked = detection(image, res_att, alpha, mark_size, v, watermark_svd)
        wat_extracted = detection(
            image, watermarked, alpha, mark_size, v, watermark_svd
        )
        # compute similarity H1
        scores.append(similarity(wat_extracted, wat_attacked))
        labels.append(1)
        # compute similarity H0
        scores.append(similarity(fakemark, wat_attacked))
        labels.append(0)
        sample += 1

# print('scores array: ', scores)
# print('labels array: ', labels)
labels_array = np.asarray(labels)
scores_array = np.asarray(scores)

# Create a mask to filter out NaN values
mask = ~np.isnan(labels_array) & ~np.isnan(scores_array)

# Apply the mask to filter out NaN values
filtered_labels = labels_array[mask]
filtered_scores = scores_array[mask]
# compute ROC
fpr, tpr, tau = roc_curve(
    np.asarray(filtered_labels), np.asarray(filtered_scores), drop_intermediate=False
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
