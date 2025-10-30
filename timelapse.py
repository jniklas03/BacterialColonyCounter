import cv2 as cv
import numpy as np
import os

from dish_detection import detect_dishes, crop
from preprocessing import preprocess, preprocess_bg_removal

def make_foreground_masks(
        image_paths,
        save_path = "",
        save = False
):
    last_image_path = image_paths[-1] # last image used for a foreground mask

    dishes, masks, coordinates, _ = detect_dishes( # dish crops from last image
        source=last_image_path,
        file_name=os.path.basename(last_image_path),
        save=False
    )

    file_name = os.path.splitext(os.path.basename(last_image_path))[0]
    foreground_masks = [preprocess(source=dish, mask=mask, file_name=file_name, kernel_size=500) for dish, mask in zip(dishes, masks)]

    if save:
        for idx, mask in enumerate(foreground_masks):
            cv.imwrite(os.path.join(save_path, f"fg_mask{idx}"), mask)

    return foreground_masks, coordinates

def make_background_masks(image_paths, n_to_stack, coordinates, save, save_path):
    first_n_image_paths = image_paths[:n_to_stack] # grabs the first n images

    first_n_dishes = []
    
    for img_path in first_n_image_paths: # crops dishes and preprocesses them
        dishes, masks = crop(img_path, coordinates)

        preprocessed = [preprocess_bg_removal(source=dish, mask=mask, file_name=os.path.splitext(os.path.basename(img_path))[0]) for dish, mask in zip(dishes, masks)]
        first_n_dishes.append(preprocessed)

    background_masks = []

    for i in range(len(first_n_dishes[0])): # combines all preprocessed images for a given dish (OR operand); results in masks of background/noise
        group = [d[i] for d in first_n_dishes]
        
        stack = np.zeros_like(group[0])
        for idx, img in enumerate(group):
            stack = cv.bitwise_or(stack, img)
            if save:
                cv.imwrite(os.path.join(save_path, f"bg_mask_dish{i+1}_{idx+1}.png"), img)
        background_masks.append(stack)
        if save:
            cv.imwrite(os.path.join(save_path, f"bg_mask_dish{i+1}_stack.png"), stack)

    return background_masks

"""
Plan für timelapse pipeline:

1. Programm scannt ne directory, sucht sich das letzte bild aus und preprocessed es mit der BIG preprocessing function => fg mask.
2. Das programm stack die masken der ersten paar bilder. Dies dient als ne maske um artefakte wie die beschriftungen oder spiegelungen zu entfernen => bg mask.
3. Es geht dann von vorne durch, runnt SMALL preprocessing, entfernt jedes mal die bg maske, overlaid die fg maske und scannt für colonies.
4. Es tracked die detected colonies, einerseits um gleich graphing und data analysis drin zu haben, andererseits, um irgendwann von SMALL preprocessing auf BIG umzusteigen.

"""