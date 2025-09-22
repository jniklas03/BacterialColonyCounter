import cv2 as cv
import numpy as np

img = cv.imread("Images/newest_image.jpg", cv.IMREAD_GRAYSCALE)
blur = cv.GaussianBlur(img, (101, 101), 0)
clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

# corrected = cv.subtract(img, blur)
# corrected = cv.normalize(corrected, None, 0, 255, cv.NORM_MINMAX)

clahe1 = clahe.apply(img)

kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (211, 211)) #around 211-221
background = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)
corrected = cv.subtract(img, background)

clahe2 = clahe.apply(corrected)

thresh = cv.adaptiveThreshold(
    corrected, 
    255, 
    cv.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv.THRESH_BINARY,
    51,
    5)


cv.imwrite("clahe1.jpg", clahe1)
cv.imwrite("clahe2.jpg", clahe2)
cv.imwrite("corrected.jpg", corrected)
cv.imwrite("background.jpg", background)
cv.imwrite("thresh.jpg", thresh)