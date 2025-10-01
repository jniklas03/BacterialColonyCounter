import cv2 as cv
import numpy as np
import os
import optuna
import yaml

def tuning(
        trial,
        metadata_path,
        preprocess_path
        ):
    data = {}

    for file in os.listdir(metadata_path):
        if file.endswith(".yaml"):
            with open(os.path.join(metadata_path, file), "r") as f:
                data[os.path.splitext(file)[0]] = yaml.safe_load(f)

    flat_data = {
        f"{date}_{key}":colonies
        for date, inner in data.items()
        for key, colonies in inner.items()
    }

    params = cv.SimpleBlobDetector_Params()
    params.minThreshold = 0
    params.maxThreshold = trial.suggest_int("maxThreshold", 0, 255) # Lower numbers = less false positives
    params.thresholdStep = trial.suggest_float("thresholdStep", 0, 50)
    params.filterByArea = True
    params.minArea = 100
    params.maxArea = 1000
    params.filterByColor = True
    params.blobColor = 0
    params.filterByCircularity = True
    params.minCircularity = trial.suggest_float("minCircularity", 0, 1)
    params.filterByConvexity = True
    params.minConvexity = trial.suggest_float("minConvexity", 0, 1)
    params.filterByInertia = True
    params.minInertiaRatio = trial.suggest_float("minInertiaRatio", 0, 1)

    detector = cv.SimpleBlobDetector_create(params)

    y_true = []
    y_pred = []

    for file_id, true_count in flat_data.items():
        pre_img_path = os.path.join(preprocess_path, f"{file_id}.jpg")

        pre_img = cv.imread(pre_img_path, cv.IMREAD_GRAYSCALE)

        blobs = detector.detect(pre_img)

        y_true.append(true_count)
        y_pred.append(len(blobs))
    
    rel_errors = [abs(p - t)/t for p, t in zip(y_pred, y_true)]
    mean_rel_error = sum(rel_errors)/len(rel_errors)
    # std_rel_error = (sum((e - mean_rel_error)**2 for e in rel_errors)/len(rel_errors))**0.5

    return mean_rel_error

metadata_path=r"C:\Users\jakub\Documents\Bachelorarbeit\Code\160925\FromPDF"
preprocess_path=r"C:\Users\jakub\Documents\Bachelorarbeit\Code\160925\Preprocessing\Training"

study = optuna.create_study(direction="minimize")

study.optimize(
    lambda trial: tuning(trial, metadata_path, preprocess_path), n_trials=50, n_jobs=-1
)

print("Best parameters:", study.best_params)
print("Best Score:", study.best_value)