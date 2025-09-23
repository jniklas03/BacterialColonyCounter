import cv2 as cv
import numpy as np

img = cv.imread("Images/newest_image.jpg", cv.IMREAD_GRAYSCALE)
blur = cv.GaussianBlur(img, (101, 101), 0)
clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (211, 211)) #around 211-221

# preprocessing var 1; clahe first, then background removal
clahe1 = clahe.apply(img)
background1 = cv.morphologyEx(clahe1, cv.MORPH_OPEN, kernel)
corrected1 = cv.subtract(clahe1, background1)
# thresh1 = cv.adaptiveThreshold(
#     corrected1, 
#     255, 
#     cv.ADAPTIVE_THRESH_GAUSSIAN_C,
#     cv.THRESH_BINARY,
#     51,
#     5)

# preprocessing var 2; background removal, then clahe
background2 = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)
corrected2 = cv.subtract(img, background2)
corrected2 = clahe.apply(corrected2)
# thresh2 = cv.adaptiveThreshold(
#     clahe2, 
#     255, 
#     cv.ADAPTIVE_THRESH_GAUSSIAN_C,
#     cv.THRESH_BINARY,
#     51,
#     5)

cv.imwrite("corr1.jpg", corrected1)
cv.imwrite("corr2.jpg", corrected2)