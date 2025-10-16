import cv2 as cv
import numpy as np
import os
import yaml
from dish_detection import detect_dishes
from preprocessing_stills import preprocess
from counting_stills import detect_colonies

def pipeline(
        source,
        n_dishes = 6,
        kernel_size=500, 
        save_path = "",
        save_metadata=False,
        save_dishes=False,
        save_preprocessed=False,
        save_detected=True
        ):
    """
    Process image to get yield cropped dishes, with circled colonies.

    Parameters
    ----------
    source: str
        Filepath to the image file of the petri dishes with the colonies.
    n_dishes: int, optional
        Amount of expected dishes.
    kernel_size: int, optional
        Kernel size for opening; higher number yields more smoothed, generally better results, but takes longer.
    save_path: str, optional
        Filepath where the images should be saved. Creates different folders for dish crops, preprocessed images, and dishes with detected colonies.
    save_metadata: bool, default=False
        Whether to save metadata - dish crop positions and number of colonies.
    save_dishes: bool, default=False
        Whether to save the dish crops.
    save_preprocessed: bool, default=False
        Whether to save the preprocessed images.
    save_detected: bool, default=True
        Whether to save the image with the detected colonies.

    """
    if not os.path.isfile(source):
        raise TypeError("source needs to be a string of a filepath.")

    img = cv.imread(source)
    file_name = os.path.splitext(os.path.basename(source))[0]

    metadata = {file_name: {}}

    dishes, dish_metadata = detect_dishes(
        source=img,
        n_dishes=n_dishes,
        save=save_dishes,
        save_path=save_path,
        file_name=file_name,
        metadata=metadata
    )
    
    preprocessed = []

    for idx, dish in enumerate(dishes):
        preprocessed.append(preprocess(
            source=dish,
            kernel_size=kernel_size,
            save=save_preprocessed,
            save_path=save_path,
            file_name=file_name,
            idx=idx+1
        ))

    for idx, preprocessed_img in enumerate(preprocessed):
        colony_metadata = detect_colonies(
            source=preprocessed_img,
            raw_img=dishes[idx],
            save=save_detected,
            save_path=save_path,
            file_name=file_name,
            idx=idx+1,
            metadata=metadata
        )

    if save_metadata:
        metadata_file = os.path.join(save_path, "metadata.yaml")
        
        if os.path.exists(metadata_file):
            with open(metadata_file, "r") as f:
                metadata = yaml.safe_load(f) or {}
        else:
            metadata = {}

        metadata.update(dish_metadata)

        with open(metadata_file, "w") as f:
            yaml.safe_dump(metadata, f)


def mult_pipeline(
        source,
        n_dishes = 6,
        kernel_size=500, 
        save_path = "",
        save_metadata=False,
        save_dishes=False,
        save_preprocessed=False,
        save_detected=True
                  ):
    for file in os.listdir(source):
        pipeline(
            source=os.path.join(source, file),
            n_dishes = n_dishes,
            kernel_size = kernel_size, 
            save_path = save_path,
            save_metadata = save_metadata,
            save_dishes = save_dishes,
            save_preprocessed=save_preprocessed,
            save_detected=save_detected
        )