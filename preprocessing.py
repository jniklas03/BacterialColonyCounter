import cv2 as cv
import numpy as np
import os

def preprocess(
        file,
        gray_img,
        save_path,
        tag = 0,
        save=False,
        kernel_size=250,
        ):
    """
    Preprocesses input file for colony detection. Returns preprocessed image.

    Keyword arguments:
    file -- Name of the file without the extension, so "22.09.2025" NOT "22.09.2025.jpg".
    gray_img -- OpenCv object of the grayscale image, i.e cv.imread(image, cv.IMREAD_GRAYSCALE).
    save_path -- Path where the preprocessed images should be saved.
    tag -- Internal parameter passed by main.py for processing and naming multiple dishes. Don't change.
    save -- Save the preprocessed images?
    kernel_size -- Kernel size for "opening"; higher number yields more smoothed, generally better results, but takes longer. Decent quick results with 250. Don't go too high with small colonies?
    """
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)) # Creates CLAHE object
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (kernel_size, kernel_size)) # Passing the kernel_size as an ellipse

    corrected = clahe.apply(gray_img) # Applies CLAHE
    background = cv.morphologyEx(corrected, cv.MORPH_OPEN, kernel) # "Opening": Smoothes away the image drastically to form the "background". 
    corrected = cv.subtract(corrected, background) # Removes the background from the image to preserve only high contrast elements (colonies).

    if save:
        save_path_preprocessing = save_path + r"\Preprocessing"
        os.makedirs(save_path_preprocessing, exist_ok=True)
        cv.imwrite(os.path.join(save_path_preprocessing, f"{file}_preprocessed_{tag+1}.jpg"), corrected)
    
    print("Preprocessing successful.")

    return(corrected)