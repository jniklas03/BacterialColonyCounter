import cv2 as cv
import numpy as np
import os
import preprocessing, colony_detection, dish_detection

def pipeline(
        raw_img_path, 
        save_path, 
        kernel_size=211,
        save_preprocessing=False,
        save_dishes=False,
        use_dishes=False,
        ):
    
    raw_img = cv.imread(raw_img_path)
    gray_img = cv.cvtColor(raw_img, cv.COLOR_BGR2GRAY)

    dishes = dish_detection.detect_dishes(
        raw_img=raw_img,
        gray_image=gray_img,
        save_path=save_path,
        save=save_dishes,
    )
    
    if use_dishes:
        pass

    else:
        preprocessed = preprocessing.preprocess(
            gray_img=gray_img,
            save_path=save_path,
            kernel_size=kernel_size,
            save=save_preprocessing
        )
    
    colony_detection.detect_colonies(
        preprocessed_img=preprocessed,
        background_img=raw_img,
        save_path=save_path
    )

pipeline(
    "SOURCE.jpg",
    r"C:\Users\jakub\Documents\Bachelorarbeit\Code\160925\Results",
    save_dishes=True,
    save_preprocessing=True
)
