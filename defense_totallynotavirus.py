def embedding(input1: str, input2: str):
    """
    Function to embed the watermark in the image

    :param input1: Name of the original image file
    :param input2: Name of the watermark file
    :return: Watermarked image
    """
    import cv2
    import numpy as np
    import pywt

    alpha = 0.3
    v = "additive"

    image = cv2.imread(input1, cv2.IMREAD_GRAYSCALE)
    mark = np.load(input2)

    LL1, (LH1, HL1, HH1) = pywt.dwt2(image, "haar")
    LL2, (LH2, HL2, HH2) = pywt.dwt2(LH1, "haar")
    LL3, (LH3, HL3, HH3) = pywt.dwt2(LL2, "haar")

    subbands = {"LH": LH3, "HL": HL3, "HH": HH3}

    watermarked_bands = []

    # Process each subband
    for _, band in subbands.items():
        # Get the locations in the subband
        sign_band = np.sign(band)
        abs_band = abs(band)
        locations_band = np.argsort(
            -abs_band, axis=None
        )  # - sign is used to get descending order
        rows_band = band.shape[0]
        locations_band = [
            (val // rows_band, val % rows_band) for val in locations_band
        ]  # locations as (x,y) coordinates

        # Embed the watermark in the subband
        watermarked_band = abs_band.copy()
        for loc, mark_val in zip(locations_band[1:], mark):
            if v == "additive":
                watermarked_band[loc] += alpha * mark_val
            elif v == "multiplicative":
                watermarked_band[loc] *= 1 + (mark_val * alpha)
        watermarked_band *= sign_band
        watermarked_bands.append(watermarked_band)

    # Restore sign and o back to spatial domain

    LL2 = pywt.idwt2(
        (LL3, (watermarked_bands[0], watermarked_bands[1], watermarked_bands[2])),
        "haar",
    )
    LH1 = pywt.idwt2((LL2, (LH2, HL2, HH2)), "haar")
    watermarked = pywt.idwt2((LL1, (LH1, HL1, HH1)), "haar")

    output1 = np.uint8(watermarked)
    return output1
