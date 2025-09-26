import cv2 as cv
import numpy as np
import os

def detect_colonies(
        file,
        preprocessed_img,
        background_img,
        save_path,
        tag = 0,
        save=True
):
    """
    
    
    """
    params = cv.SimpleBlobDetector_Params()
    params.minThreshold = 0
    params.maxThreshold = 255
    params.thresholdStep = 10
    params.filterByArea = True
    params.minArea = 100
    params.maxArea = 1000
    params.filterByColor = True
    params.blobColor = 0
    params.filterByCircularity = True
    params.minCircularity = 0.3
    params.filterByConvexity = False
    # params.minConvexity = 0.9
    params.filterByInertia = True
    params.minInertiaRatio = 0.2

    detector = cv.SimpleBlobDetector_create(params)

    blobs = detector.detect(preprocessed_img)

    output = cv.drawKeypoints(background_img, blobs, np.array([]), (0,255,0), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    if save:
        save_path_blob_detection = save_path + r"\Colonies"
        os.makedirs(save_path_blob_detection, exist_ok=True)
        cv.imwrite(os.path.join(save_path_blob_detection, f"{file}_colonies_{tag+1}.jpg"), output)
        with open(f"{os.path.join(save_path_blob_detection, file)}.txt", "a") as fp:
            fp.write(f"{len(blobs)} \n")

    print(f"{len(blobs)} colonies detected.")

    return(len(blobs))