PATH = r"C:\Users\jakub\Documents\Bachelorarbeit\Resources\Sources\27.10.2025"
IMG = r"C:\Users\jakub\Documents\Bachelorarbeit\Resources\Sources\22.09.2025.jpg"
SAVE = r"C:\Users\jakub\Documents\Bachelorarbeit\Resources\Timelapse_test"

import phase

phase.timelapse_pipeline(
    PATH, 
    True,
    SAVE, 
    plot=True
)