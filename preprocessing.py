import cv2 as cv
import numpy as np
import os
import warnings
from inputs import read_img
import random

from skimage.feature import peak_local_max
from scipy import ndimage as ndi
from skimage.segmentation import watershed as sk_watershed

def watershed(
        threshold,
        kernel_size=3,
        min_distance=10
):
    # --- Load and pre-process ---
    th = cv.imread(threshold, cv.IMREAD_GRAYSCALE)
    if th is None:
        raise ValueError(f"Could not read image: {threshold}")

    # Smooth a bit to reduce noise
    th_blur = cv.GaussianBlur(th, (5, 5), 0)

    # Binary mask (Otsu)
    _, binary = cv.threshold(th_blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    # Distance transform
    dist = cv.distanceTransform(binary, cv.DIST_L2, 5)

    # Local maxima for watershed seeds
    coords = peak_local_max(dist, min_distance=min_distance, labels=binary)

    local_maxi = np.zeros_like(dist, dtype=bool)
    local_maxi[tuple(coords.T)] = True
    markers, _ = ndi.label(local_maxi)

    # Watershed segmentation
    labels = sk_watershed(-dist, markers, mask=binary)

    # --- Random color fill visualization ---
    colored = np.zeros((th.shape[0], th.shape[1], 3), dtype=np.uint8)
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels != 0]  # skip background

    rng = np.random.default_rng(42)  # reproducible colors
    random_colors = (rng.uniform(50, 255, size=(len(unique_labels), 3))).astype(np.uint8)

    for label_val, color in zip(unique_labels, random_colors):
        colored[labels == label_val] = color

    # --- Binary mask with thin black outlines ---
    separated_binary = np.zeros_like(th, dtype=np.uint8)
    separated_binary[labels > 0] = 255

    # Draw black contours for each colony
    for label_val in unique_labels:
        mask = np.uint8(labels == label_val)
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cv.drawContours(separated_binary, contours, -1, 0, 1)  # black (0), thickness=1

    return colored, separated_binary

def preprocess(
        source,
        mask = None,
        kernel_size = 200,
        save=False,
        save_path = "",
        file_name = "preprocessed",
        idx: int = None
        ):
    """
    Preprocesses input image for colony detection. Returns preprocessed image.
    
    Parameters
    ----------
    source: str or np.ndarray
        Image of dish, or string of to the image path
    mask: np.ndarray, optional
        Mask of background area outside of dish, passed by detect_dishes()
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

    green_channel = img[:, :, 1]

    blur = cv.medianBlur(green_channel, 5)

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (kernel_size, kernel_size))

    tophat = cv.morphologyEx(blur, cv.MORPH_TOPHAT, kernel)

    _, threshold = cv.threshold(tophat, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

    if mask is not None:
        threshold = cv.bitwise_and(threshold, threshold, mask=mask)

    save_name = f"{file_name}_p{idx}.png" if idx is not None else f"{file_name}_p.png"

    if save:
        save_path_preprocessing = os.path.join(save_path, "Preprocessing")
        os.makedirs(save_path_preprocessing, exist_ok=True)
        cv.imwrite(os.path.join(save_path_preprocessing, save_name), threshold)
    print(f"File {save_name} preprocessed.")

    return threshold

def preprocess_small(
        source,
        kernel_size = 21,
        s = 121,
        C = 11,
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

    green_channel = img[:, :, 1]

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (kernel_size, kernel_size))

    blackhat = cv.morphologyEx(green_channel, cv.MORPH_BLACKHAT, kernel)

    th = cv.adaptiveThreshold(
        src=blackhat,
        maxValue=255,
        adaptiveMethod=cv.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv.THRESH_BINARY_INV,
        blockSize=s,
        C=C
    )
    save_name = f"{file_name}_p{idx}.png" if idx is not None else f"{file_name}_p.png"

    if save:
        save_path_preprocessing = os.path.join(save_path, "Preprocessing")
        os.makedirs(save_path_preprocessing, exist_ok=True)
        cv.imwrite(os.path.join(save_path_preprocessing, save_name), th)
    print(f"File {save_name} preprocessed.")

    return th