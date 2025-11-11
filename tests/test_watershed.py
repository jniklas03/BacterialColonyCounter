from phase.image_manipulation.preprocessing import separate_components

IMG = r"C:\Users\jakub\Documents\Bachelorarbeit\Resources\Timelapse_test\Preprocessing\28.10.2025-13.00.02_preprocessed_1.png"
DISH = r"C:\Users\jakub\Documents\Bachelorarbeit\Resources\Timelapse_test\Dishes\28.10.2025-13.00.02_dish_1.jpg"
SAVE = r"C:\Users\jakub\Documents\Bachelorarbeit\Resources\Timelapse_test"

separate_components(DISH, IMG, max_dist=0.65)
