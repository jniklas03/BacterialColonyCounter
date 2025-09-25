import cv2 as cv
import numpy as np
import os
import preprocessing, colony_detection, dish_detection

def pipeline(
        source, 
        save_path, 
        kernel_size=261, #221 for whole image
        save_preprocessing=False,
        save_dishes=False,
        ):
    
    file = os.path.splitext(os.path.basename(source))[0]
    raw_img = cv.imread(source)
    gray_img = cv.cvtColor(raw_img, cv.COLOR_BGR2GRAY)

    dishes = dish_detection.detect_dishes(
        file=file,
        raw_img=raw_img,
        gray_image=gray_img,
        save_path=save_path,
        save=save_dishes,
    )
    
    preprocessed = []

    for idx, dish in enumerate(dishes):
        gray_dish = cv.cvtColor(dish, cv.COLOR_BGR2GRAY)
        preprocessed.append(preprocessing.preprocess(
            file=file,
            gray_img=gray_dish,
            save_path=save_path,
            kernel_size=kernel_size,
            save=save_preprocessing,
            tag=idx
        ))

    for idx, pre in enumerate(preprocessed):
        colony_detection.detect_colonies(
            file=file,
            preprocessed_img=pre,
            background_img=dishes[idx],
            save_path=save_path,
            tag=idx
        )

pipeline(
    "Results/Sources/22.09.2025.jpg",
    "Results"
)

