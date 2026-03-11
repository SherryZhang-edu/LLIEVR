import numpy as np
from tensorflow import lite
from PIL import Image
import os
from utils.enhancer import  enhance_gamma, enhance_he, enhance_clahe
import cv2
from utils.evaluation import load_darkzurich_rgb_list

source_image_dir = 'datasets\\RealData\\video1_indoor'
output_image_dir = 'experiments\\results\\RealData'

image_list = load_darkzurich_rgb_list(source_image_dir)
for img_path in image_list:
    img = cv2.imread(img_path)
    
    dataset_name = os.path.basename(source_image_dir)
    
    # clahe enhancement
    enhanced_img = enhance_clahe(img)
    output_path = os.path.join(output_image_dir, dataset_name+'_clahe', os.path.basename(img_path))
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, enhanced_img)
    
    # he enhancement
    enhanced_img = enhance_he(img)
    output_path = os.path.join(output_image_dir, dataset_name+'_he', os.path.basename(img_path))
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, enhanced_img)
    
    # gamma enhancement
    gamma = 4
    enhanced_img = enhance_gamma(img, gamma=gamma)
    output_path = os.path.join(output_image_dir, dataset_name+'_gamma_'+str(gamma), os.path.basename(img_path))
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, enhanced_img)

    