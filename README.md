# Catch the mark challenge - "totallynotavirus" group

These are all the files needed for the challenge.

## Attack

The attack file contains all the code to attack a given image with one or more attacks.

## Defense

Contains the embedding function, which receives in input the image and the watermark to embed.

## Detection

Here the watermark is extracted from the given images.
We rely on an hardcoded SVD value to better distingiush the mark in the image.

## Roc estimaton

This file can be executed alone, and estimates a suitable value as treshold for the similaritty.

## Demo

Runs the whole process, from embedding to detection on all ./img/\*.bmp with all the attacks.
