import cv2 as cv
import numpy as np
import os
import preprocessing, colony_detection, dish_detection

def pipeline(
        raw_img_path, 
        save_path, 
        kernel_size=251, #221 for whole image
        save_preprocessing=False,
        save_dishes=False,
        ):
    
    raw_img = cv.imread(raw_img_path)
    gray_img = cv.cvtColor(raw_img, cv.COLOR_BGR2GRAY)

    dishes = dish_detection.detect_dishes(
        raw_img=raw_img,
        gray_image=gray_img,
        save_path=save_path,
        save=save_dishes,
    )
    
    preprocessed = []

    for idx, dish in enumerate(dishes):
        gray_dish = cv.cvtColor(dish, cv.COLOR_BGR2GRAY)
        preprocessed.append(preprocessing.preprocess(
            gray_img=gray_dish,
            save_path=save_path,
            kernel_size=kernel_size,
            save=save_preprocessing,
            tag=idx
        ))

    for idx, pre in enumerate(preprocessed):
        colony_detection.detect_colonies(
            preprocessed_img=pre,
            background_img=dishes[idx],
            save_path=save_path,
            tag=idx
        )

pipeline(
    "Results/SOURCE.jpg",
    "Results",
    save_preprocessing=True,
    save_dishes=True
)