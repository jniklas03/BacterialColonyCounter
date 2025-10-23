import cv2 as cv
import numpy as np
import os
import warnings
from inputs import read_img

import cv2 as cv
import numpy as np
import random

def watershed(
        threshold,
        kernel_size=3,
        fg_thresh_ratio=0.4  # lower = more segments (e.g. 0.4–0.6)
):
    # Read grayscale image
    th = cv.imread(threshold, cv.IMREAD_GRAYSCALE)
    if th is None:
        raise ValueError("Could not read image.")

    # Smooth a bit to remove noise
    th_blur = cv.medianBlur(th, 5)

    # Morphological background estimation
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (kernel_size, kernel_size))
    sure_bg = cv.dilate(th_blur, kernel, iterations=3)

    # Distance transform
    dist_transform = cv.distanceTransform(th_blur, cv.DIST_L2, 5)

    # Threshold for sure foreground
    _, sure_fg = cv.threshold(dist_transform, fg_thresh_ratio * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    # Unknown region (border)
    unknown = cv.subtract(sure_bg, sure_fg)

    # Marker labelling
    _, markers = cv.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    # Convert grayscale to color
    th_color = cv.cvtColor(th, cv.COLOR_GRAY2BGR)

    # Apply watershed
    cv.watershed(th_color, markers)

    # Generate random color map for each region
    unique_labels = np.unique(markers)
    colors = {label: [random.randint(50, 255) for _ in range(3)] for label in unique_labels if label > 1}

    # Create colored result
    colored = np.zeros_like(th_color)
    for label, color in colors.items():
        colored[markers == label] = color

    # Draw green outlines where watershed borders are (-1)
    colored[markers == -1] = [0, 255, 0]

    return colored, markers

def preprocess(
        source,
        kernel_size = 500,
        save=False,
        save_path = "",
        file_name = "preprocessed",
        idx: int = None
        ):
    """
    Preprocesses input image for colony detection. Returns preprocessed image.
    
    Parameters
    ----------
    source: str or np.ndarray or list
        Image of dish, or string of to the image path, or list of images.
    kernel_size: int, optional
        Kernel size for "opening"; higher number yields more smoothed, generally better results, but takes longer. 
    save: bool, default=True
        Whether to save the cropped dishes.
    save_path: str, optional
        Path to directory where the image is saved.
    idx: int, optional
        Passed by main.py if multiple dishes processed.

    Returns
    -------
    np.ndarray
        Preprocessed dish.
    """
    img = read_img(source=source)

    if save and not save_path:
        warnings.warn(f"No specified save path. Images saved in the current directory ({os.getcwd()}) under ...Preprocessing.")

    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    _, mask = cv.threshold(gray_img, 0, 1, cv.THRESH_BINARY)  # everything except pure black

    green_channel = img[:, :, 1]

    blur = cv.medianBlur(green_channel, 5)

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (kernel_size, kernel_size))

    tophat = cv.morphologyEx(blur, cv.MORPH_TOPHAT, kernel)
    
    tophat_masked = cv.bitwise_and(tophat, tophat, mask=mask)

    _, threshold = cv.threshold(tophat_masked, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    threshold = cv.bitwise_not(threshold)
    threshold[mask == 0] = 0

    save_name = f"{file_name}_p{idx}.jpg" if idx is not None else f"{file_name}_p.jpg"

    if save:
        save_path_preprocessing = os.path.join(save_path, "Preprocessing")
        os.makedirs(save_path_preprocessing, exist_ok=True)
        cv.imwrite(os.path.join(save_path_preprocessing, save_name), threshold)
    print(f"File {save_name} preprocessed.")

    return threshold

# def preprocess_small(
#         source,
#         kernel_size = 21,
#         s = 121,
#         C = 11,
#         save=False,
#         save_path = "",
#         file_name = "preprocessed",
#         idx: int = None
#         ):
#     """
#     Preprocesses input image for colony detection. Returns preprocessed image.
    
#     Parameters
#     ----------
#     source: str or np.ndarray or list
#         Image of dish, or string of to the image path, or list of images.
#     kernel_size: int, optional
#         Kernel size for "opening"; higher number yields more smoothed, generally better results, but takes longer. 
#     save: bool, default=True
#         Whether to save the cropped dishes.
#     save_path: str, optional
#         Path to directory where the image is saved.
#     idx: int, optional
#         Passed by main.py if multiple dishes processed.

#     Returns
#     -------
#     np.ndarray
#         Preprocessed dish.
#     """
#     img = read_img(source=source)

#     if save and not save_path:
#         warnings.warn(f"No specified save path. Images saved in the current directory ({os.getcwd()}) under ...Preprocessing.")

#     green_channel = img[:, :, 1]

#     kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (kernel_size, kernel_size))

#     blackhat = cv.morphologyEx(green_channel, cv.MORPH_BLACKHAT, kernel)

#     th = cv.adaptiveThreshold(
#         src=blackhat,
#         maxValue=255,
#         adaptiveMethod=cv.ADAPTIVE_THRESH_GAUSSIAN_C,
#         thresholdType=cv.THRESH_BINARY_INV,
#         blockSize=s,
#         C=C
#     )
#     save_name = f"{file_name}_p{idx}.jpg" if idx is not None else f"{file_name}_p.jpg"

#     if save:
#         save_path_preprocessing = os.path.join(save_path, "Preprocessing")
#         os.makedirs(save_path_preprocessing, exist_ok=True)
#         cv.imwrite(os.path.join(save_path_preprocessing, save_name), th)
#     print(f"File {save_name} preprocessed.")

#     return th