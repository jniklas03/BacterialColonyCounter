import cv2 as cv
import numpy as np
import os
import yaml
import warnings, logging
from inputs import read_img
from dish_detection import detect_dishes
from preprocessing import preprocess
from colony_detection import detect_colonies

def pipeline(
        source,
        n_dishes = 6,
        kernel_size=500, 
        save_path = "",
        save_metadata=False,
        save_metadata_coordinates=False,
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
    save_metadata_coordinates: bool, default=False
        Whether to save colony coordinates in the metadata file.
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

    dishes, data = detect_dishes(
        source=img,
        n_dishes=n_dishes,
        save=save_dishes,
        save_metadata=save_metadata,
        save_path=save_path,
        file_name=file_name
    )
    
    preprocessed = []

    for dish in dishes:
        preprocessed.append(preprocess(
            source=dish,
            kernel_size=kernel_size,
            save=save_preprocessed,
            save_path=save_path
        ))

    for idx, preprocessed_img in enumerate(preprocessed):
        detect_colonies(
            source=preprocessed_img,
            raw_img=dishes[idx],
            save=save_detected,
            save_path=save_path,
            save_metadata=save_metadata,
            metadata=data,
            save_coordinates=save_metadata_coordinates,
            file_name=file_name
        )

# def mult_pipeline(
#         file_path,
#         save_path,
#         kernel_size=250, 
#         save_preprocessing=False,
#         save_dishes=False,
#         n_dishes = 6,
#         save_yaml = False
#                   ):
#     for file in os.listdir(file_path):
#         pipeline(
#             source=os.path.join(file_path, file),
#             save_path=save_path,
#             kernel_size=kernel_size,
#             save_preprocessing=save_preprocessing,
#             save_dishes=save_dishes,
#             n_dishes=n_dishes,
#             save_yaml=save_yaml
#         )

pipeline(
    source=r"C:\Users\jakub\Documents\Bachelorarbeit\Resources\Sources\01.10.2025\02.10.2025-02.30.03.jpg",
    save_path=r"C:\Users\jakub\Documents\Bachelorarbeit\Resources\Detection_test",
    save_detected=True,
    save_dishes=True,
    save_metadata=True,
    save_metadata_coordinates=True,
    save_preprocessed=True
)