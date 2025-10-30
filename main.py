import cv2 as cv
import numpy as np
import os
import yaml
from inputs import read_time, read_image_paths
from dish_detection import detect_dishes, crop
from preprocessing import preprocess, preprocess_small
from counting import detect_colonies
from timelapse import *

def pipeline(
        source,
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

    dishes, masks, dish_metadata = detect_dishes(
        source=img,
        save=save_dishes,
        save_path=save_path,
        file_name=file_name,
        metadata=metadata
    )
    
    preprocessed = []

    for idx, dish in enumerate(dishes):
        preprocessed.append(preprocess(
            source=dish,
            mask=masks[idx],
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

def timelapse_pipeline(
        source,
        save,
        save_path = "",
        n_to_stack=5
):
    image_paths, file_names = read_image_paths(source)

    fg_masks, coordinates = make_foreground_masks(image_paths=image_paths, save_path=save_path, save=False)

    bg_masks = make_background_masks(image_paths=image_paths, n_to_stack=n_to_stack, coordinates=coordinates, save_path=save_path, save=False)
    
    dish_counts = {i: [] for i in range(len(coordinates))}
    
    for img_path, file_name in zip(image_paths, file_names):
        timestamp = read_time(img_path)

        dishes, masks = crop(img_path, coordinates)
        preprocessed_imgs = []

        for idx, (dish, mask) in enumerate(zip(dishes, masks)):
            preprocessed = preprocess_small(source=dish, mask=mask, file_name=file_name, save=False, save_path=save_path)

            mask_applied = cv.bitwise_and(preprocessed, preprocessed, mask=fg_masks[idx])
            if save:
                cv.imwrite(os.path.join(save_path, f"{file_name}_mask1_dish{idx+1}.png"), mask_applied)

            mask_applied = cv.bitwise_and(mask_applied, mask_applied, mask=cv.bitwise_not(bg_masks[idx]))
            if save:
                cv.imwrite(os.path.join(save_path, f"{file_name}_mask2_dish{idx+1}.png"), mask_applied)

            preprocessed_imgs.append(mask_applied)

        for idx, img in enumerate(preprocessed_imgs):
            count, _, _ = detect_colonies(source=img, save=True, save_path=save_path, file_name=os.path.splitext(os.path.basename(img_path))[0], idx=idx+1)

            dish_counts[idx].append((timestamp, count))

    return(dish_counts)