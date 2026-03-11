import numpy as np
from tensorflow import lite
from PIL import Image
import os
from utils.enhancer import load_model, enhance_gamma, enhance_he, enhance_clahe, enhance_image_cv
import cv2

model_path = 'lolv1.tflite'  
# input_image_path = 'datasets\\lle\\darkZurich\\set1\\GP010376_frame_000297_rgb_anon.png'  
# input_image_path = 'datasets\\lle\\LOLdataset\\eval15\\low\\79.png'
# input_image_path = "datasets\\RealData\\video4_office\\frame_0005.png"
input_image_path = "datasets\\RealData\\video2_walk\\frame_0015.png"

# 1. 加载模型（只需加载一次）
interpreter = load_model(model_path)

# 2. 调用函数进行增强
try:
    # # 你可以直接传入路径
    # enhanced_img = enhance_image_PIL(input_image_path)
    
    # # 或者传入 PIL Image 对象
    # # img_obj = Image.open(input_image_path)
    # # enhanced_img = enhance_image(img_obj)
    
    # # 3. 显示或保存结果
    # # enhanced_img.show()
    # enhanced_img.save('enhanced_result.png')
    # print("图像增强成功并已保存！")
    img_cv = cv2.imread(input_image_path)
    img_cv = cv2.resize(img_cv, (600, 400))
    
    # use mobileIE to enhance
    enhanced_cv = enhance_image_cv(img_cv,interpreter)
    cv2.imwrite('enhanced_result_cv.png', enhanced_cv)
    print("MobielIE done！")
    
    # use gamma correction to enhance
    enhanced_gamma = enhance_gamma(img_cv, gamma=4)
    cv2.imwrite('enhanced_gamma.png', enhanced_gamma)
    print("Gamma correction done！")
    
    # use HE to enhance
    enhanced_he = enhance_he(img_cv)
    cv2.imwrite('enhanced_he.png', enhanced_he)
    print("HE done！")
    
    # use CLAHE to enhance
    enhanced_clahe = enhance_clahe(img_cv)
    cv2.imwrite('enhanced_clahe.png', enhanced_clahe)
    print("CLAHE done！")
    
    # show comparison
    # top_row = np.hstack((img_cv, enhanced_cv))
    # bottom_row = np.hstack((enhanced_he, enhanced_clahe,enhanced_gamma))
    # combined = np.vstack((top_row, bottom_row))
    
    combined = np.hstack((img_cv, enhanced_cv, enhanced_gamma, enhanced_he, enhanced_clahe))
    cv2.imshow('Original | MobileIE | Gamma | HE | CLAHE', combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
except Exception as e:
    print(f"处理出错: {e}")