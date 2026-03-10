# resize the data into (600,400) for a fair comparison, and also to speed up the evaluation
import cv2
import numpy as np
import os

def resize_image_cv(image_bgr, target_size=(600, 400)):
    return cv2.resize(image_bgr, target_size, interpolation=cv2.INTER_CUBIC)

img_dir = "datasets/lle/darkZurich/set1"
save_dir = "datasets/lle/darkZurich/set1_resized"
os.makedirs(save_dir, exist_ok=True)
for img_name in os.listdir(img_dir):
    if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue
    img_path = os.path.join(img_dir, img_name)
    img = cv2.imread(img_path)
    resized_img = resize_image_cv(img, target_size=(600, 400))
    save_path = os.path.join(save_dir, img_name)
    cv2.imwrite(save_path, resized_img)

