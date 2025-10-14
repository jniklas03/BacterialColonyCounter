import cv2 as cv
import numpy as np
import os
import yaml
import warnings
from inputs import read_img

def detect_dishes(
        source,
        n_dishes=6, 
        save=True,
        save_path = "",
        file_name = "dish_detected", 
        metadata: dict = None
):
    """
    Detects dishes in the image and crops in around them. Returns the cropped images as a list.
    
    Parameters
    ----------
    source: str or np.ndarray
        Raw image, or string of to the image path.
    n_dishes: int, default=6
        Amount of expected dishes.
    save: bool, default=True
        Whether to save the cropped dishes.
    save_path: str, optional
        Path to directory where the images and metadata are saved.
    file_name: str, optional
        Name to save the dishes as.
    metadata: dict, optional
        Metadata dictionary handled by main.py.

    Returns
    -------
    list of np.ndarray
        List of cropped dishes.
    dict
        Metadata
    """
    
    img = read_img(source=source)
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    if save and not save_path:
        warnings.warn(f"No specified save path. Images saved in the current directory ({os.getcwd()}) under ...Dishes.")

    save_path_dish_detection = os.path.join(save_path, "Dishes") # path for the dish crops

    circles = cv.HoughCircles( # creates a numpy array of detected circles
        gray_img, # image, should be grayscale
        cv.HOUGH_GRADIENT, # detection method
        dp=1.2, # resolution used for the detection; dp=2 means half resolution of the original image
        minDist=900, # minimum distance between the centers of circles in px
        param1=100, # upper threshold for canny edge detection (uses canny edge detection internally)
        param2=30, # threshold for center detection; lower values are more sensitive, but prone to false positves
        minRadius=400, # minimum and maxmimum radius in px; both set very generously due to problems with detection
        maxRadius=1000 
    )

    dishes = [] # list for the cropped images

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int") 

        circles = circles[:n_dishes] # limits junk output to the expected amount of dishes; the dishes should appear in the first crops

        for idx, (x, y, r) in enumerate(circles, start=1): # circles defined by their x and y coordinates of their center as well as their radius; idx is used for naming the files

            mask = np.zeros_like(gray_img) # mask (black image) the size of the input image
            cv.circle(mask, (x, y), r, 255, -1) # fills the mask with white circles at the location of the detected dishes

            masked_img = cv.bitwise_and(  # applies mask (keeps values where the mask is white) to original image
                img, 
                img, 
                mask=mask)

            x1, y1 = max(0, x-r), max(0, y-r) # defines top left corner of the crop
            x2, y2 = min(img.shape[1], x+r), min(img.shape[0], y+r) # defines bottom right corner of the crop
            square_crop = masked_img[y1:y2, x1:x2] # applies a square crop for the masked dishes

            dishes.append(square_crop)

            if metadata:
                metadata[file_name][idx] = [
                    {
                        "center": [int(x), int(y)],
                        "radius": int(r),
                        "colony_count": None,
                        # "colony_characteristics": []
                    }]

            save_name = f"{file_name}_d{idx}.jpg" if idx is not None else f"{file_name}_d.jpg"

            if save: # saving the dishes if the flag is passed
                os.makedirs(save_path_dish_detection, exist_ok=True)
                cv.imwrite(os.path.join(save_path_dish_detection, save_name), square_crop)

        print(f"{len(circles)} dishes detected in file: {file_name}.")

    else:
        warnings.warn("No dishes detected.")

    return dishes, metadata
