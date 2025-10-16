import cv2 as cv
import numpy as np
import os
import warnings
from inputs import read_img

def preprocess_t(
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
    save_name = f"{file_name}_p{idx}.jpg" if idx is not None else f"{file_name}_p.jpg"

    if save:
        save_path_preprocessing = os.path.join(save_path, "Preprocessing")
        os.makedirs(save_path_preprocessing, exist_ok=True)
        cv.imwrite(os.path.join(save_path_preprocessing, save_name), th)
    print(f"File {save_name} preprocessed.")

    return th