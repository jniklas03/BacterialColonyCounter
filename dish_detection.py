import cv2 as cv
import numpy as np
import os

def detect_dishes(data, file, raw_img, gray_img, save_path, n_dishes=6, save=True):
    """
    Detects dishes in the image and crops in around them. Returns the cropped images as a list.
    
    Keyword arguments:
    data -- data.yaml metadata file.
    file -- Name of the file without the extension, so "22.09.2025" NOT "22.09.2025.jpg".
    raw_img -- OpenCv object of the raw image, i.e. cv.imread(image).
    gray_img -- OpenCv object of the grayscale image, i.e cv.imread(image, cv.IMREAD_GRAYSCALE).
    save_path -- Path where the cropped dishes should be saved.
    n_dishes -- Amount of expected dishes. If a dish isn't detected turn it up. WILL detect random stuff if set higher than necessary.
    save -- Save the crops of the dishes?
    """
    
    save_path_dish_detection = save_path + r"\Dishes" # path for the dish crops

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

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int") 

        circles = circles[:n_dishes] # limits junk output to the expected amount of dishes; the dishes should appear in the first crops
        dishes = [] # list for the cropped images

        for idx, (x, y, r) in enumerate(circles, start=1): # circles defined by their x and y coordinates of their center as well as their radius; idx is used for naming the files

            mask = np.zeros_like(gray_img) # mask (black image) the size of the input image
            cv.circle(mask, (x, y), r, 255, -1) # fills the mask with white circles at the location of the detected dishes

            masked_img = cv.bitwise_and(  # applies mask (keeps values where the mask is white) to original image
                raw_img, 
                raw_img, 
                mask=mask)

            x1, y1 = max(0, x-r), max(0, y-r) # defines top left corner of the crop
            x2, y2 = min(raw_img.shape[1], x+r), min(raw_img.shape[0], y+r) # defines bottom right corner of the crop
            square_crop = masked_img[y1:y2, x1:x2] # applies a square crop for the masked dishes

            dishes.append(square_crop)

            if data:
                data[file]["dishes"].append(
                    {
                        "id":idx,
                        "center": [int(x), int(y)],
                        "radius": int(r),
                        "colony_count": None
                    }
                )

            if save: # saving the dishes if the flag is passed
                os.makedirs(save_path_dish_detection, exist_ok=True)
                cv.imwrite(os.path.join(save_path_dish_detection, f"{file}_dish_{idx}.png"), square_crop)

        print(f"{len(circles)} dishes detected.")

        
    else:
        print("No circles detected.")

    return(dishes, data)
