import numpy as np
from tensorflow import lite
from PIL import Image
import os
import cv2
import time
from utils.enhancer import enhance_image_cv, enhance_gamma, enhance_he, enhance_clahe

# ===============================
# 单线程设置（很重要）
# ===============================
os.environ["OMP_NUM_THREADS"] = "1"
cv2.setNumThreads(1)

# ===============================
# ORB 初始化
# ===============================
orb = cv2.ORB_create(nfeatures=1500)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

# ===============================
# 计时 pipeline
# ===============================
def run_pipeline(img1, img2, method_name, enhance_func=None, repeat=3):

    T_enh_list = []
    T_feat_list = []
    T_match_list = []
    T_ransac_list = []

    for _ in range(repeat):

        # -------- Enhancement --------
        if enhance_func is not None:
            start = time.perf_counter()
            img1_proc = enhance_func(img1)
            img2_proc = enhance_func(img2)
            T_enh = time.perf_counter() - start
        else:
            img1_proc = img1
            img2_proc = img2
            T_enh = 0.0

        # -------- Feature --------
        start = time.perf_counter()
        kp1, des1 = orb.detectAndCompute(img1_proc, None)
        kp2, des2 = orb.detectAndCompute(img2_proc, None)
        T_feat = time.perf_counter() - start

        # -------- Matching --------
        start = time.perf_counter()
        matches = bf.knnMatch(des1, des2, k=2)
        good = [m for m, n in matches if m.distance < 0.75 * n.distance]
        T_match = time.perf_counter() - start

        # -------- RANSAC --------
        start = time.perf_counter()
        if len(good) >= 4:
            pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
            pts2 = np.float32([kp2[m.trainIdx].pt for m in good])
            H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 3.0)
        T_ransac = time.perf_counter() - start

        T_enh_list.append(T_enh)
        T_feat_list.append(T_feat)
        T_match_list.append(T_match)
        T_ransac_list.append(T_ransac)

    # 平均
    T_enh = np.mean(T_enh_list)
    T_feat = np.mean(T_feat_list)
    T_match = np.mean(T_match_list)
    T_ransac = np.mean(T_ransac_list)
    T_total = T_enh + T_feat + T_match + T_ransac

    print(f"\n===== {method_name} =====")
    print(f"Enhancement : {T_enh*1000:.2f} ms")
    print(f"Feature     : {T_feat*1000:.2f} ms")
    print(f"Matching    : {T_match*1000:.2f} ms")
    print(f"RANSAC      : {T_ransac*1000:.2f} ms")
    print("--------------------------------")
    print(f"Total       : {T_total*1000:.2f} ms")
    print(f"FPS         : {1.0/T_total:.2f}")
    print(f"Enh Ratio   : {T_enh/T_total:.2f}")

    return T_total


# ===============================
# 主程序
# ===============================
if __name__ == "__main__":

    model_path = 'lolv1.tflite'
    input_image_path1 = 'datasets\\RealData\\video2_walk\\frame_0040.png'
    input_image_path2 = 'datasets\\RealData\\video2_walk\\frame_0041.png'

    interpreter = lite.Interpreter(model_path=model_path, num_threads=4 )
    interpreter.allocate_tensors()

    # 读取两张连续帧（示例用同一张）
    img1 = cv2.imread(input_image_path1)
    img1 = cv2.resize(img1, (600, 400))
    img2 = cv2.imread(input_image_path2)
    img2 = cv2.resize(img2, (600, 400))

    # ===============================
    # 预热（非常重要）
    # ===============================
    print("Warming up...")

    dummy = np.random.randint(0,255,(600,400,3),dtype=np.uint8)

    for _ in range(10):
        enhance_image_cv(dummy, interpreter)

    for _ in range(10):
        orb.detectAndCompute(dummy, None)

    print("Warm-up done.")

    # ===============================
    # 各方法测试
    # ===============================

    run_pipeline(img1, img2, "Original", enhance_func=None)

    run_pipeline(img1, img2, "MobileIE",
                 enhance_func=lambda x: enhance_image_cv(x, interpreter))

    run_pipeline(img1, img2, "Gamma",
                 enhance_func=lambda x: enhance_gamma(x, gamma=4))

    run_pipeline(img1, img2, "HE",
                 enhance_func=enhance_he)

    run_pipeline(img1, img2, "CLAHE",
                 enhance_func=enhance_clahe)