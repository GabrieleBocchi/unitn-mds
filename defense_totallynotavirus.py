def embedding(input1: str, input2: str):
    """
    Function to embed the watermark in the image

    :param input1: Name of the original image file
    :param input2: Name of the watermark file
    :return: Watermarked image
    """
    from itertools import cycle

    import cv2
    import numpy as np
    import pywt

    ALPHAS = {
        "1": 0.03,
        "2": 0.02,
        "3": 0.01,
        "4": 0.005,
    }

    image = cv2.imread(input1, cv2.IMREAD_GRAYSCALE)
    mark = np.load(input2)

    LL, (LH, HL, HH) = pywt.dwt2(image, "haar")
    LL_LL, (LL_LH, LL_HL, LL_HH) = pywt.dwt2(LL, "haar")
    LL__LL_LL, (LL_LL_LH, LL_LL_HL, LL_LL_HH) = pywt.dwt2(LL_LL, "haar")

    subbands = {
        "LH": (LH, "1"),
        "HL": (HL, "1"),
        "HH": (HH, "1"),
        "LL_LH": (LL_LH, "2"),
        "LL_HL": (LL_HL, "2"),
        "LL_HH": (LL_HH, "2"),
        "LL_LL_LH": (LL_LL_LH, "3"),
        "LL_LL_HL": (LL_LL_HL, "3"),
        "LL_LL_HH": (LL_LL_HH, "3"),
        "LL_LL_LL": (LL__LL_LL, "4"),
    }

    active = [
        "LH",
        "HL",
        "HH",
        "LL_LH",
        "LL_HL",
        "LL_HH",
        "LL_LL_LH",
        "LL_LL_HL",
        "LL_LL_HH",
        "LL_LL_LL",
    ]

    watermarked_bands = {}

    for band_name, (band, depth) in subbands.items():
        abs_band = abs(band).flatten()

        position_value = [
            (i, abs_band[i]) for i in range(len(abs_band)) if abs_band[i] > 1
        ]
        position_value.sort(key=lambda x: x[1], reverse=True)
        locations = [
            (
                position_value[i][0] // band.shape[0],
                position_value[i][0] % band.shape[1],
            )
            for i in range(len(position_value))
        ]

        if len(locations) < len(mark) or band_name not in active:
            watermarked_bands[band_name] = band.copy()
            continue

        c_mark = cycle(mark)
        watermarked_band = band.copy()
        for loc in locations:
            watermarked_band[loc] += ALPHAS[depth] * next(c_mark)
        watermarked_bands[band_name] = watermarked_band

    LL_LL = pywt.idwt2(
        (
            watermarked_bands["LL_LL_LL"],
            (
                watermarked_bands["LL_LL_LH"],
                watermarked_bands["LL_LL_HL"],
                watermarked_bands["LL_LL_HH"],
            ),
        ),
        "haar",
    )

    LL = pywt.idwt2(
        (
            LL_LL,
            (
                watermarked_bands["LL_LH"],
                watermarked_bands["LL_HL"],
                watermarked_bands["LL_HH"],
            ),
        ),
        "haar",
    )

    watermarked = pywt.idwt2(
        (
            LL,
            (
                watermarked_bands["LH"],
                watermarked_bands["HL"],
                watermarked_bands["HH"],
            ),
        ),
        "haar",
    )

    output1 = np.uint8(watermarked)
    return output1
