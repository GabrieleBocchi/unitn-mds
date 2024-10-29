# Catch the mark challenge - "totallynotavirus" group

These are all the files needed for the challenge.

## Attack

The attack file contains all the code to attack a given image with one or more attacks.

To attack you can call the function 'attacks', with the following parameters:

- input1: Filename of the watermarked image
- attack_name: list of attacks to perform (possible values: awgn, blur, sharpen, median, resize, jpeg)
- param_array: parameters for the attacks if needed (sigma, alpha)

Returns the attacked image

## Defense

Contains the embedding function, which receives in input the image and the watermark to embed.

To embed the watermark, use the 'embedding' function with the following parameters:

- input1: Path for the original image
- input2: Path for the watermark

Returns the watermarked image

## Detection

Here the watermark is extracted from the given images.
We rely on an hardcoded SVD value to better distinguish the mark in the image.

Use the 'detection' function with the following parameters:

- input1: Path of the original image
- input2: Path of the watermarked image
- input3: Path of the attacked image

Returns (1 if the similarity is greater than the threshold, wpsnr)

## Roc estimaton

This file can be executed alone, and estimates a suitable value as treshold for the similaritty.

- Generates the roc curve

## Demo

Runs the whole process, from embedding to detection on all ./img/\*.bmp with all the attacks.
