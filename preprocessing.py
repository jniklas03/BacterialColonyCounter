import cv2 as cv
import numpy as np
import os

def preprocess(
        gray_img,
        save_path,
        tag = 0,
        save=False,
        kernel_size=211,
        ):
  
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (kernel_size, kernel_size)) #around 211-221
    background = cv.morphologyEx(gray_img, cv.MORPH_OPEN, kernel)
    corrected = cv.subtract(gray_img, background)
    corrected = clahe.apply(corrected)

    if save:
        save_path_preprocessing = save_path + r"\Preprocessing"
        os.makedirs(save_path_preprocessing, exist_ok=True)
        cv.imwrite(os.path.join(save_path_preprocessing, f"preprocessed_{tag}.jpg"), corrected)
    
    print("Preprocessing successful.")

    return(corrected)