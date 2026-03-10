import numpy as np
from tensorflow import lite
from PIL import Image
import os
import cv2

def load_model(model_path):
    """
    加载 TFLite 模型
    :param model_path: 模型文件路径
    :return: interpreter 对象
    """
    interpreter = lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

def enhance_image_PIL(image, interpreter, target_size=(600, 400)):
    """
    输入一张图片，输出增强后的图片
    :param image: 可以是图片路径 (str)，也可以是 PIL.Image 对象
    :param interpreter: TFLite 模型解释器
    :param target_size: 目标图像尺寸
    :return: 增强后的 PIL.Image 对象
    """    
    # 1. 读取并转换图像
    if isinstance(image, str):
        if not os.path.exists(image):
            raise FileNotFoundError(f"找不到图片: {image}")
        img = Image.open(image).convert("RGB")
    elif isinstance(image, Image.Image):
        img = image.convert("RGB")
    else:
        raise ValueError("输入必须是图片路径或 PIL.Image 对象")

    # 2. 图像预处理
    img = img.resize(target_size, Image.BICUBIC)
    img_array = np.array(img).astype(np.float32) / 255.0  
    img_array = np.expand_dims(img_array, axis=0)  


    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # 注意：如果你的 TFLite 模型支持动态输入尺寸，且输入图片尺寸不固定，
    # 可能需要取消下面两行的注释来动态调整输入张量大小：
    # interpreter.resize_tensor_input(input_details[0]['index'], img_array.shape)
    # interpreter.allocate_tensors()

    # 3. 推理
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    output_img = interpreter.get_tensor(output_details[0]['index'])
    
    # 4. 后处理
    output_img = np.clip(output_img, 0., 1.)
    output_img = np.squeeze(output_img)
    output_img = (output_img * 255).astype(np.uint8)  

    # 5. 返回 PIL Image 对象
    return Image.fromarray(output_img)

# for OpenCV 用户的增强函数
def enhance_image_cv(image_bgr, interpreter, target_size=(600, 400)):
    """
    输入一张 OpenCV 格式的图片，输出增强后的 OpenCV 格式图片
    :param image_bgr: OpenCV 读取的图像 (numpy.ndarray, BGR 格式)
    :param interpreter: TFLite 模型解释器
    :param target_size: 目标图像尺寸
    :return: 增强后的图像 (numpy.ndarray, BGR 格式)
    """
        
    if not isinstance(image_bgr, np.ndarray):
        raise ValueError("输入必须是 OpenCV 图像 (numpy.ndarray)")

    if (image_bgr.shape[1], image_bgr.shape[0]) != target_size:
        image_bgr = cv2.resize(image_bgr, target_size, interpolation=cv2.INTER_CUBIC)
    # 1. BGR 转 RGB (OpenCV 默认是 BGR，而模型通常需要 RGB)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    # 2. 图像预处理：归一化并增加 Batch 维度
    img_array = image_rgb.astype(np.float32) / 255.0  
    img_array = np.expand_dims(img_array, axis=0)  

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # 注意：如果你的 TFLite 模型支持动态输入尺寸，且输入图片尺寸不固定，
    # 可能需要取消下面两行的注释来动态调整输入张量大小：
    # interpreter.resize_tensor_input(input_details[0]['index'], img_array.shape)
    # interpreter.allocate_tensors()

    # 3. 推理
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    output_img = interpreter.get_tensor(output_details[0]['index'])
    
    # 4. 后处理：限制范围、去掉 Batch 维度、反归一化
    output_img = np.clip(output_img, 0., 1.)
    output_img = np.squeeze(output_img)
    output_img = (output_img * 255).astype(np.uint8)  

    # 5. RGB 转回 BGR，以便继续使用 OpenCV 处理或保存
    enhanced_bgr = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)

    return enhanced_bgr


def enhance_gamma(image_bgr, gamma=0.4):
    """
    2. Gamma Correction (伽马校正)
    通过非线性变换提升暗部细节。对于暗光增强，gamma 值通常小于 1 (如 0.3 - 0.6)。
    :param image_bgr: 输入图像 (BGR)
    :param gamma: 伽马值
    :return: 增强后的图像 (BGR)
    """
    # 构建查找表 (Look-Up Table, LUT) 以加速计算
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 
                      for i in np.arange(0, 256)]).astype("uint8")
    
    # 应用查找表
    enhanced_bgr = cv2.LUT(image_bgr, table)
    return enhanced_bgr

def enhance_he(image_bgr):
    """
    3. HE (Histogram Equalization - 全局直方图均衡化)
    将图像转换到 LAB 色彩空间，仅对 L (亮度) 通道进行均衡化，避免颜色偏移。
    :param image_bgr: 输入图像 (BGR)
    :return: 增强后的图像 (BGR)
    """
    # BGR 转 LAB
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # 对 L 通道应用全局直方图均衡化
    l_eq = cv2.equalizeHist(l)
    
    # 合并通道并转回 BGR
    lab_eq = cv2.merge((l_eq, a, b))
    enhanced_bgr = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
    
    return enhanced_bgr

def enhance_clahe(image_bgr, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    4. CLAHE (Contrast Limited Adaptive Histogram Equalization - 限制对比度自适应直方图均衡化)
    相比 HE，CLAHE 将图像分成小块(tiles)进行均衡化，并限制对比度以防止过度放大噪声。
    :param image_bgr: 输入图像 (BGR)
    :param clip_limit: 对比度限制阈值
    :param tile_grid_size: 网格大小
    :return: 增强后的图像 (BGR)
    """
    # BGR 转 LAB
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # 创建 CLAHE 对象并应用于 L 通道
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_clahe = clahe.apply(l)
    
    # 合并通道并转回 BGR
    lab_clahe = cv2.merge((l_clahe, a, b))
    enhanced_bgr = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
    
    return enhanced_bgr

# ================= 使用示例 =================
if __name__ == "__main__":
    model_path = 'lolv1.tflite'  
    # input_image_path = 'datasets\\lle\\darkZurich\\set1\\GP010376_frame_000297_rgb_anon.png'  
    input_image_path = 'datasets\\lle\\LOLdataset\\eval15\\low\\665.png'
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
        enhanced_gamma = enhance_gamma(img_cv, gamma=0.4)
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