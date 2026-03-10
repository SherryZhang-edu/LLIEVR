import cv2
import numpy as np
from utils.enhancer import enhance_image_cv,load_model
from utils.evaluation import load_darkzurich_rgb_list
import os

model_path = 'lolv1.tflite'
interpreter = load_model(model_path)
def enhance(img):
    return enhance_image_cv(img, interpreter)

sequence_path = "datasets/lle/darkZurich/set1_resized"  # 替换成你的 TUM 数据集路径
ouptut_path = "experiments/results/darkZurich/set1_enhanced"  # 替换成你想保存增强图像的路径
os.makedirs(ouptut_path, exist_ok=True)

image_list = load_darkzurich_rgb_list(sequence_path)

print("Total frames:", len(image_list))

image_list = image_list[:10]

for img_path in image_list:
    img = cv2.imread(img_path)
    enhanced_img = enhance(img)

    # # 显示原图和增强后的图像
    # combined = np.hstack((img, enhanced_img))
    # cv2.imshow('Original (Left) vs Enhanced (Right)', combined)
    # cv2.waitKey(0)
    
    # save enhanced image
    filename = os.path.basename(img_path)
    save_path = os.path.join(ouptut_path, filename)
    cv2.imwrite(save_path, enhanced_img)
print(f"Enhanced image saved to: {ouptut_path}")
    