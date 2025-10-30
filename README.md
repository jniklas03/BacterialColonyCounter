# Persistent Counter
Python script based on openCV used for detection and counting of persistent bacterial colonies.

## About
The project contains:
**A dish cropping module**
<img width="4056" height="3040" alt="dish_detected_debug" src="https://github.com/user-attachments/assets/5459a217-162e-4092-af31-9a14cd70b894" />
**A preprocessing module**
<img width="1040" height="1040" alt="28 10 2025-02 00 02_mask2_dish4" src="https://github.com/user-attachments/assets/39db4c1c-7370-4382-9f6a-924519ca8868" />
**A counting module**
<img width="1040" height="1040" alt="28 10 2025-02 00 02_c4" src="https://github.com/user-attachments/assets/f4e45113-0196-4569-b631-484d56130389" />

as well as various helper modules.


## Usage
The script can be used to crop, preprocess, and count the colonies on single images with `pipeline()`, or entire directories with `mult_pipeline()`. Finally, it can be used on growth timelapses with `timelapse_pipeline()`.
