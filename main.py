import cv2 as cv
import numpy as np
import os
import preprocessing, colony_detection, dish_detection
import yaml

def pipeline(
        source,
        save_path,
        kernel_size=500, 
        save_preprocessing=False,
        save_dishes=False,
        n_dishes = 6,
        save_yaml = False
        ):
    """
    Process image to get yield cropped dishes, with circled colonies.

    Keyword arguments:
    source -- Filepath to the image file of the petri dishes with the colonies.
    save_path -- Filepath where the images should be saved. The script makes different folders for different images by default.
    kernel_size -- Kernel size for opening; higher number yields more smoothed, generally better results, but takes longer. Decent quick results with 250.
    save_preprocessing -- Save preprocessed dishes?
    save_dishes -- Save unprocessed cropped dishes?
    n_dishes -- Amount of expected dishes. If a dish isn't detected turn it up. WILL detect random stuff if set higher than necessary.
    """
    file = os.path.splitext(os.path.basename(source))[0]
    raw_img = cv.imread(source)
    gray_img = cv.cvtColor(raw_img, cv.COLOR_BGR2GRAY)

    data = {}
    data[file] = {"dishes":[]}

    dishes, data = dish_detection.detect_dishes(
        data=data,
        file=file,
        raw_img=raw_img,
        gray_img=gray_img,
        save_path=save_path,
        save=save_dishes,
        n_dishes=n_dishes
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
            data=data,
            file=file,
            preprocessed_img=pre,
            raw_img=dishes[idx],
            save_path=save_path,
            tag=idx
        )

    if save_yaml:
        with open(os.path.join(save_path, "data.yaml"), "w") as f:
            yaml.dump(data, f)


def mult_pipeline(
        file_path,
        save_path,
        kernel_size=250, 
        save_preprocessing=False,
        save_dishes=False,
        n_dishes = 6,
        save_yaml = False
                  ):
    for file in os.listdir(file_path):
        pipeline(
            source=os.path.join(file_path, file),
            save_path=save_path,
            kernel_size=kernel_size,
            save_preprocessing=save_preprocessing,
            save_dishes=save_dishes,
            n_dishes=n_dishes,
            save_yaml=save_yaml
        )

    
mult_pipeline(
    r"C:\Users\jakub\Documents\Bachelorarbeit\Code\160925\Sources\01.10.2025",
    r"C:\Users\jakub\Documents\Bachelorarbeit\Code\160925\Detection_test"
    )
