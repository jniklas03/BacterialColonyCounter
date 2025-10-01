import cv2 as cv
import numpy as np
import os

def detect_colonies(
        data,
        file,
        preprocessed_img,
        raw_img,
        save_path,
        tag = 0,
        save=True
):
    """
    Detects individual colonies. 

    Keyword arguments:
    data -- data.yaml metadata file.
    file -- Name of the file without the extension, so "22.09.2025" NOT "22.09.2025.jpg".
    preprocessed_img -- openCV object of preprocessed image (grayscale).
    raw_img -- openCV object of raw/initial image.
    save_path -- Filepath where the images should be saved. The script makes different folders for different images by default.
    tag -- Internal parameter passed by main.py for processing and naming multiple dishes. Don't change.
    save -- Save the preprocessed images?
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

    output = cv.drawKeypoints(raw_img, blobs, np.array([]), (0,255,0), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    if data:
        dish = data[file]["dishes"][tag]
        dish["colony_count"] = len(blobs)
        dish["colonies"] = [{"x": int(k.pt[0]), "y": int(k.pt[1]), "size": k.size} for k in blobs]


    if save:
        save_path_blob_detection = save_path + r"\Colonies"
        os.makedirs(save_path_blob_detection, exist_ok=True)
        cv.imwrite(os.path.join(save_path_blob_detection, f"{file}_colonies_{tag+1}.jpg"), output)
        # with open(f"{os.path.join(save_path_blob_detection, file)}.txt", "a") as fp:
        #     fp.write(f"{len(blobs)} \n")

    print(f"{len(blobs)} colonies detected.")

    return(data)