import cv2 as cv
import numpy as np

img = cv.imread('im160925crop.jpg')

hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

lower_red = np.array([1, 50, 51]) # dye red
upper_red = np.array([10, 163, 108])

mask = cv.inRange(hsv, lower_red, upper_red)

res = cv.bitwise_and(img, img, mask=mask)

cv.imwrite("mask.png", mask)
cv.imwrite("res.png", res)

kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2,2))
mask_clean = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel, iterations=2)
mask_clean = cv.morphologyEx(mask_clean, cv.MORPH_DILATE, kernel, iterations=1)

cv.imwrite("mask_clean.png", mask_clean)

contours, _ = cv.findContours(mask_clean, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

print("Colonies counted:", len(contours))

