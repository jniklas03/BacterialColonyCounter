import cv2 as cv
import numpy as np
import os
import warnings

from inputs import read_img
from dish_detection import detect_dishes, crop

def preprocess(
        source,
        mask,
        area_filter = True,
        s = 121,
        C = 11,
        kernel_size = 3,
        min_area = 5,
        max_area = 200,
        save=False,
        save_path = "",
        file_name = "preprocessed",
        idx: int = None
        ):
    """
    Preprocesses input image. Used in the preprocessing of just grown colonies (timelapse).
    Returns preprocessed image.
    
    Parameters
    ----------
    source: str or np.ndarray
        Image of dish, or string of to the image path
    mask: np.ndarray, optional
        Mask of background area outside of dish
    s: int, default=121
        Block size for thresholding. Bigger numbers include more to threshold.
    C: int, defualt=11
        Constant to subtract from thresholding.
    kernel_size: int, optional
        Kernel size for erosion. Used for noise removal.
    save: bool, default=True
        Whether to save the cropped dishes.
    save_path: str, optional
        Path to directory where the image is saved.
    file_name: str, optional
        Name to save the preprocessed image as.
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

    threshold = cv.adaptiveThreshold(
        src=green_channel,
        maxValue=255,
        adaptiveMethod=cv.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv.THRESH_BINARY_INV,
        blockSize=s,
        C=C
    )

    if area_filter:
        num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(threshold, connectivity=8)
        filtered = np.zeros_like(threshold)

        for i in range(1, num_labels):  # skip background
            area = stats[i, cv.CC_STAT_AREA]
            if min_area <= area <= max_area:
                filtered[labels == i] = 255
    else:
        filtered = threshold

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (kernel_size, kernel_size))
    eroded = cv.morphologyEx(filtered, cv.MORPH_ERODE, kernel)

    if mask is not None:
        filtered = cv.bitwise_and(filtered, filtered, mask=mask)
    else:
        filtered = eroded

    save_name = f"{file_name}_p{idx}.png" if idx is not None else f"{file_name}_p.png"

    if save:
        save_path_preprocessing = os.path.join(save_path, "Preprocessing")
        os.makedirs(save_path_preprocessing, exist_ok=True)
        cv.imwrite(os.path.join(save_path_preprocessing, save_name), filtered)
    print(f"File {save_name} preprocessed.")

    return filtered

def preprocess_fg_isolation(
        source,
        mask = None,
        kernel_size = 200,
        save=False,
        save_path = "",
        file_name = "preprocessed",
        idx: int = None
        ):
    """
    Preprocesses input image. Used for standalone colony detection or for foreground masking.
    Returns preprocessed image.
    
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
    file_name: str, optional
        Name to save the preprocessed image as.
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

def preprocess_bg_isolation(
        source,
        mask,
        s = 121,
        C = 11,
        kernel_size = 5,
        min_area = 5,
        max_area = 200,
        save=False,
        save_path = "",
        file_name = "preprocessed",
        idx: int = None
        ):
    """
    Preprocesses input image. Basically inverse of preprocess_small. Used for removal of petri dish and other artefacts.
    Returns preprocessed image.
    
    Parameters
    ----------
    source: str or np.ndarray
        Image of dish, or string of to the image path
    mask: np.ndarray, optional
        Mask of background area outside of dish
    s: int, default=121
        Block size for thresholding. Bigger numbers include more to threshold.
    C: int, defualt=11
        Constant to subtract from thresholding.
    kernel_size: int, optional
        Kernel size for erosion. Used for noise removal.
    save: bool, default=True
        Whether to save the cropped dishes.
    save_path: str, optional
        Path to directory where the image is saved.
    file_name: str, optional
        Name to save the preprocessed image as.
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

    threshold = cv.adaptiveThreshold(
        src=green_channel,
        maxValue=255,
        adaptiveMethod=cv.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv.THRESH_BINARY_INV,
        blockSize=s,
        C=C
    )

    num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(threshold, connectivity=8)
    filtered = np.zeros_like(threshold)

    for i in range(1, num_labels):  # skip background
        area = stats[i, cv.CC_STAT_AREA]
        if not min_area <= area <= max_area:
            filtered[labels == i] = 255

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (kernel_size, kernel_size))
    opened = cv.morphologyEx(filtered, cv.MORPH_OPEN, kernel, iterations=2)

    if mask is not None:
        filtered = cv.bitwise_and(filtered, filtered, mask=mask)
    else:
        filtered = opened

    save_name = f"{file_name}_p{idx}.png" if idx is not None else f"{file_name}_p.png"

    if save:
        save_path_preprocessing = os.path.join(save_path, "Preprocessing")
        os.makedirs(save_path_preprocessing, exist_ok=True)
        cv.imwrite(os.path.join(save_path_preprocessing, save_name), filtered)
    print(f"File {save_name} preprocessed.")

    return filtered