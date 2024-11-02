# Catch the mark challenge - "totallynotavirus" group

This competition is designed by the Professor Giulia Boato and is a mandatory activity for the course Multimedia Data Security.
The concepts of the challenge is to apply watermarking concepts seen during theoretical lectures in a practical way.

## Timeline

---

Each group is asked to develop it's own code and submit the work some days before the challenge.
The day of the challenge activities are divided in two phases:

- **defense phase**, where each group need to embed a watermark inside three images with the chosen strategy
- **attack phase**, where each group is asked to perform a variety of attacks on images watermarked by others groups.
  The goal is to remove the watermark of as much images as possible without degrading the image quality

Rules and details are better explained in the `BattleRules2024.pdf` file.
The repo contains all the files needed for the challenge.

Below we are going to briefly explain the main files.

## Attack

---

The attack file contains all the code to attack a given image with one or more attacks.

To attack you can call the function `attacks`, with the following parameters:

- **input1**: Filename of the watermarked image
- **attack_name**: list of attacks to perform (possible values: awgn, blur, sharpen, median, resize, jpeg)
- **param_array**: parameters for the attacks if needed (sigma, alpha)

Returns the attacked image

## Defense

---

Contains the embedding function, which receives in input the image and the watermark to embed.

Our strategy is to apply **Discrete Wavelet Transform (DWT)** three times to a given image.
The watermark is injected in all HL, LH, HH bands found during the DWT decomposition until layer #3.
After some investigations we decided to use an _additive_ approach with a pretty low value of _aplha_.

Then the image is reconstructed by reversing the transform operations.

To embed the watermark, use the `embedding` function with the following parameters:

- **input1**: Path for the original image
- **input2**: Path for the watermark

The function returns the watermarked image.

## Detection

---

Here the watermark is extracted from the given images.
The same procedure applied in the `embedding` phase is applied to each of the three images.
After extracting watermarks from watermarked and attacked images if the similarity is greater than the one provided by the ROC curve, the mark is considered recovered.

Use the `detection` function with the following parameters:

- **input1**: Path of the original image
- **input2**: Path of the watermarked image
- **input3**: Path of the attacked image

Returns: a _binary value_ indicating whether the watermark is present or not (1 if the similarity is greater than the threshold)
and the _WPSNR_ value between the attacked and watermarked images.

## Roc estimaton

---

This file can be executed alone, and estimates a suitable value as threshold for the similarity.
This contains most of the functions developed in other parts of the repo, since the ROC computation needs to perform embedding, attacking and detection operations on a set of images.
For each image (contained in `img` folder) perform all the processing phases and register the result.
These can be correct decisions (_true positive_, _true negative_), or bad ones (_false positive_, _false negative_).

The computed data is used to generates the ROC curve.

## Demo

---

Runs the whole process, from embedding to detection on all ./img/\*.bmp with all the attacks.
