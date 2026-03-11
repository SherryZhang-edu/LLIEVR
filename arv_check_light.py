#check the brightness for the video

from utils.evaluation import load_darkzurich_rgb_list
import cv2
import numpy as np

sequence_path  = "datasets\\RealData\\video1_indoor"
image_list = load_darkzurich_rgb_list(sequence_path)
for img_pth in image_list:
    img = cv2.imread(img_pth)                
    gray_temp = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mu = np.mean(gray_temp)
    # match the file name with the brightness
    print(f"{img_pth}: brightness={mu}")
    