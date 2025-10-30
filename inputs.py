import cv2 as cv, numpy as np, os
import re
from datetime import datetime

def read_img(source):
    """
    Reads and returns image from a string filepath or a numpy array.
    """
    if isinstance(source, str) and os.path.isfile(source):
        img = cv.imread(source)
    elif isinstance(source, np.ndarray):
        img = source
    else:
        raise TypeError("source must be a file path or a NumPy array or a list")
    return img

def read_time(filename):
    """
    Extracts datetime from image filenames, i.e. '01.10.2025-17.40.02.jpg'
    Returns datetime object, or None if pattern not found.
    """
    # get just the filename, remove extension
    basename = os.path.basename(filename)
    name, _ = os.path.splitext(basename)

    # pattern: DD.MM.YYYY-HH.MM.SS
    match = re.match(r"(\d{2})\.(\d{2})\.(\d{4})-(\d{2})\.(\d{2})\.(\d{2})", name)
    if not match:
        return None

    day, month, year, hour, minute, second = map(int, match.groups())
    return datetime(year, month, day, hour, minute, second)

def read_image_paths(source):
    if not isinstance(source, str) or not os.path.isdir(source):
        raise TypeError("source_directory must be a directory path.")
    
    image_paths = []
    base_names = []

    for img in sorted(os.listdir(source)):
        full_path = os.path.join(source, img)
        if os.path.isfile(full_path):
            image_paths.append(full_path)
            base_names.append(os.path.splitext(img)[0])

    return image_paths, base_names