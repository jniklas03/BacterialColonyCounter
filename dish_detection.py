import cv2 as cv
import numpy as np
import os

def detect_dishes(file, raw_img, gray_image, save_path, save=True):
    save_path_dish_detection = save_path + r"\Dishes"

    circles = cv.HoughCircles(
        gray_image,
        cv.HOUGH_GRADIENT,
        dp=1.2,
        minDist=900,
        param1=100,
        param2=30,
        minRadius=400,
        maxRadius=1000
    )

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")

        # keep only the first 6 (keeps detecting random stuff if slice > n_dishes)
        circles = circles[:6]
        dishes = []

        for i, (x, y, r) in enumerate(circles, start=1):

            mask = np.zeros_like(gray_image)
            cv.circle(mask, (x, y), r, 255, -1)

            dish_crop = cv.bitwise_and(raw_img, raw_img, mask=mask)

            x1, y1 = max(0, x-r), max(0, y-r)
            x2, y2 = min(raw_img.shape[1], x+r), min(raw_img.shape[0], y+r)
            square_crop = dish_crop[y1:y2, x1:x2]

            dishes.append(square_crop)

            if save:
                os.makedirs(save_path_dish_detection, exist_ok=True)
                cv.imwrite(os.path.join(save_path_dish_detection, f"{file}_dish_{i}.png"), square_crop)

        print(f"{len(circles)} dishes detected.")

        
    else:
        print("No circles detected.")

    return(dishes)
