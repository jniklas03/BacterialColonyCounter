import cv2 as cv, numpy as np, os

def read_img(source):
    if isinstance(source, str) and os.path.isfile(source):
        img = cv.imread(source)
    elif isinstance(source, np.ndarray):
        img = source
    else:
        raise TypeError("source must be a file path or a NumPy array or a list")
    return img