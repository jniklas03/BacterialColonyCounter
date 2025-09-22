import cv2 as cv
import numpy as np

# first image - rpi; otsus binarisation
img1 = cv.imread('images/newest_image_crop_unguenstig.jpg', cv.IMREAD_GRAYSCALE)
cv.imwrite("gray1.png", img1)
blur1 = cv.GaussianBlur(img1, (5,5), 0)
ret1, th1 = cv.threshold(blur1,255,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
cv.imwrite("otsu1.png", th1)

#second image - colony counter; otsus binarisation
img2 = cv.imread('images/WT1_P1_counts193_img1.png', cv.IMREAD_GRAYSCALE)
cv.imwrite("gray2.png", img2)
ret2, th2 = cv.threshold(img2,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
cv.imwrite("otsu2.png", th2)

# noise removal
kernel = np.ones((3,3), np.uint8)
opening = cv.morphologyEx(th2,cv.MORPH_OPEN, kernel, iterations = 2)

# sure background area
sure_bg = cv.dilate(opening,kernel,iterations=3)
cv.imwrite("surebg.png", sure_bg)

 
# Finding sure foreground area
dist_transform = cv.distanceTransform(opening,cv.DIST_L2,5)
ret, sure_fg = cv.threshold(dist_transform,0.7*dist_transform.max(),255,0)
 
# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv.subtract(sure_bg,sure_fg)