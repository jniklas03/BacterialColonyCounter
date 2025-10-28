import cv2 as cv
import numpy as np
import os
import warnings
from inputs import read_img

def detect_colonies(
        source,
        raw_img=None,
        save=True,
        save_path = "",
        file_name = "colonies_detected",
        metadata: dict = None,
        idx: int = None
):
    """
    Detects individual colonies. 

    Parameters
    ----------
    source: str or np.ndarray
        Preprocessed image (grayscale) or string to the image path.
    raw_img: np.ndarray, optional
        Initial image.
    save: bool, default=False
        Whether to save the image with detected colonies.
    save_path: str, optional
        Path to directory where the image and metadata are saved.
    file_name: str, optional
        File name for the image with the detected colonies.
    metadata: dict, optional
        Metadata dictionary handled by main.py.
    idx: int, optional
        Passed by main.py if multiple dishes processed.

    Returns
    -------
    dict
        Metadata
    """
    img = read_img(source=source)

    if raw_img is None:
        raw_img = img

    if save and not save_path:
        warnings.warn(f"No specified save path. Images saved in the current directory ({os.getcwd()}) under ...Colonies.")

    params = cv.SimpleBlobDetector_Params() # Values from hyperparameter tuning

    params.minThreshold = 0
    params.maxThreshold = 96 # Smaller values = less false positives
    params.thresholdStep = 5.3943585868546915 # Smaller values = more true positives

    params.filterByArea = True # Area in pxs
    params.minArea = 100 # Generous values
    params.maxArea = 1000

    params.filterByColor = True
    params.blobColor = 255 # Accepts dark/black colonies after preprocessing

    params.filterByCircularity = True
    params.minCircularity = 0.20895557940856835

    params.filterByConvexity = True
    params.minConvexity = 0.8019905165503702

    params.filterByInertia = True
    params.minInertiaRatio = 0.00610705929990651

    detector = cv.SimpleBlobDetector_create(params) # Creates detector object
    blobs = detector.detect(img) # Blobs are markers around colonies

    output = cv.drawKeypoints(raw_img, blobs, np.array([]), (0,255,0), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS) # Output = initial image with colonies marked

    if metadata is not None and idx is not None:
        metadata[file_name][idx][0]["colony_count"] = len(blobs)


    save_name = f"{file_name}_c{idx}.png" if idx is not None else f"{file_name}_c.png"

    if save: # Saving images with marked colonies
        save_path_blob_detection = os.path.join(save_path, "Colonies")
        os.makedirs(save_path_blob_detection, exist_ok=True)
        cv.imwrite(os.path.join(save_path_blob_detection, save_name), output)

    print(f"{len(blobs)} colonies detected in file {save_name}.")

    return len(blobs), metadata