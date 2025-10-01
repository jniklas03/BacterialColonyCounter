import cv2 as cv
import numpy as np
import os

def detect_colonies(
        file,
        preprocessed_img,
        raw_img,
        save_path,
        tag = 0,
        save=True,
        data=None
):
    """
    Detects individual colonies. 

    Keyword arguments:
    file -- Name of the file without the extension, so "22.09.2025" NOT "22.09.2025.jpg".
    preprocessed_img -- openCV object of preprocessed image (grayscale).
    raw_img -- openCV object of raw/initial image.
    save_path -- Filepath where the images should be saved. The script makes different folders for different images by default.
    tag -- Internal parameter passed by main.py for processing and naming multiple dishes. Don't change.
    save -- Save the preprocessed images?
    data -- data.yaml metadata file.
    """
    params = cv.SimpleBlobDetector_Params() # Values from hyperparameter tuning

    params.minThreshold = 0
    params.maxThreshold = 64 # Smaller values = less false positives
    params.thresholdStep = 2.975050229896966 # Smaller values = more true positives

    params.filterByArea = True # Area in pxs
    params.minArea = 100 # Generous values
    params.maxArea = 1000

    params.filterByColor = True
    params.blobColor = 0 # Accepts dark/black colonies after preprocessing

    params.filterByCircularity = True
    params.minCircularity = 0.41063466457046227

    params.filterByConvexity = False
    params.minConvexity = 0.7208306409286382

    params.filterByInertia = True
    params.minInertiaRatio = 0.11002280443338333

    detector = cv.SimpleBlobDetector_create(params) # Creates detector object
    blobs = detector.detect(preprocessed_img) # Blobs are markers around colonies

    output = cv.drawKeypoints(raw_img, blobs, np.array([]), (0,255,0), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS) # Output = initial image with colonies marked

    if data: # If metadata file present (by running from main.py) saves colony count and optionally colony cooridnates
        dish = data[file]["dishes"][tag]
        dish["colony_count"] = len(blobs)
        # dish["colonies"] = [{"x": int(blob.pt[0]), "y": int(blob.pt[1]), "size": round(blob.size, 3)} for blob in blobs]


    if save: # Saving images with marked colonies
        save_path_blob_detection = save_path + r"\Colonies"
        os.makedirs(save_path_blob_detection, exist_ok=True)
        cv.imwrite(os.path.join(save_path_blob_detection, f"{file}_colonies_{tag+1}.jpg"), output)

    print(f"{len(blobs)} colonies detected.")

    return(data)

# detect_colonies(
#     file=os.path.splitext(os.path.basename("Sources/23.09.2025.jpg"))[0],
#     preprocessed_img=cv.imread("Preprocessing/23.09.2025_preprocessed_1.jpg"),
#     raw_img=cv.imread("Dishes/23.09.2025_dish_1.png"),
#     save_path=r"C:\Users\jakub\Documents\Bachelorarbeit\Code\160925",
#     save=False
# )