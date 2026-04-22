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
# Adaptive trigger 阈值 (match paper Eq. 1)
# ===============================
TAU_MU = 60.0      # brightness threshold
TAU_I  = 30        # inlier threshold


# ===============================
# 计时 pipeline (固定增强 / 无增强)
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
        T_ransac = 0.0
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
    T_enh    = np.mean(T_enh_list)
    T_feat   = np.mean(T_feat_list)
    T_match  = np.mean(T_match_list)
    T_ransac = np.mean(T_ransac_list)
    T_total  = T_enh + T_feat + T_match + T_ransac

    print(f"\n===== {method_name} =====")
    print(f"Trigger     :   0.00 ms   (n/a)")
    print(f"Enhancement : {T_enh*1000:7.2f} ms")
    print(f"Feature     : {T_feat*1000:7.2f} ms")
    print(f"Matching    : {T_match*1000:7.2f} ms")
    print(f"RANSAC      : {T_ransac*1000:7.2f} ms")
    print("--------------------------------")
    print(f"Total       : {T_total*1000:7.2f} ms")
    print(f"FPS         : {1.0/T_total:7.2f}")
    print(f"Enh Ratio   : {T_enh/T_total:7.2f}")

    return T_total


# ===============================
# 计时 pipeline (Adaptive Trigger)
# ===============================
def run_pipeline_adaptive(img1, img2, method_name,
                          enhance_func,
                          tau_mu=TAU_MU, tau_I=TAU_I,
                          prev_inliers_init=100,
                          repeat=3):
    """
    Implements Eq. (1):
        g_t = (mu_t < tau_mu)  OR  (I_{t-1} < tau_I)
    Profiles trigger cost separately and records whether the frame
    was routed through Enhance path or Bypass path.
    """

    T_trig_list   = []
    T_enh_list    = []
    T_feat_list   = []
    T_match_list  = []
    T_ransac_list = []

    # 分桶统计 triggered vs bypassed frame
    trig_hits   = 0   # frames that went through enhancement
    trig_total  = 0   # total frames processed (= 2 * repeat)

    # triggered / bypassed 分别的总耗时
    total_triggered_ms = []
    total_bypassed_ms  = []

    prev_inliers = prev_inliers_init

    for _ in range(repeat):

        frame_timings = []  # collect per-frame total (2 frames per repeat)

        processed_frames = []  # (gray_or_bgr, used_enhanced)

        # -------------- 逐帧处理两帧 --------------
        for img in (img1, img2):

            # ---------- Trigger ----------
            t0 = time.perf_counter()
            gray_temp = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            mu = cv2.mean(gray_temp)[0]                  # faster than np.mean
            g_t = (mu < tau_mu) or (prev_inliers < tau_I)
            T_trig = time.perf_counter() - t0

            # ---------- Enhancement (conditional) ----------
            t0 = time.perf_counter()
            if g_t:
                img_enh = enhance_func(img)
                # ORB needs grayscale; pay the re-cvtColor on enhanced path
                gray_for_orb = cv2.cvtColor(img_enh, cv2.COLOR_BGR2GRAY)
                used_enhanced = True
            else:
                # Bypass path: reuse gray_temp, no extra cvtColor
                gray_for_orb = gray_temp
                used_enhanced = False
            T_enh = time.perf_counter() - t0

            # ---------- Feature ----------
            t0 = time.perf_counter()
            kp, des = orb.detectAndCompute(gray_for_orb, None)
            T_feat = time.perf_counter() - t0

            T_trig_list.append(T_trig)
            T_enh_list.append(T_enh)
            T_feat_list.append(T_feat)

            processed_frames.append((kp, des))

            # bookkeeping
            trig_total += 1
            if used_enhanced:
                trig_hits += 1

            frame_timings.append((T_trig, T_enh, T_feat, used_enhanced))

        # ---------- Matching (pair of frames) ----------
        (kp1, des1), (kp2, des2) = processed_frames

        t0 = time.perf_counter()
        matches = bf.knnMatch(des1, des2, k=2)
        good = [m for m, n in matches if m.distance < 0.75 * n.distance]
        T_match = time.perf_counter() - t0
        T_match_list.append(T_match)

        # ---------- RANSAC ----------
        T_ransac = 0.0
        inliers = 0
        t0 = time.perf_counter()
        if len(good) >= 4:
            pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
            pts2 = np.float32([kp2[m.trainIdx].pt for m in good])
            H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 3.0)
            if mask is not None:
                inliers = int(mask.sum())
        T_ransac = time.perf_counter() - t0
        T_ransac_list.append(T_ransac)

        # feedback 给下一轮 (Eq. 1 的 I_{t-1})
        prev_inliers = inliers if inliers > 0 else prev_inliers

        # 把 matching / ransac 时间均摊到 2 帧里来计算 per-frame total
        per_frame_match  = T_match  / 2.0
        per_frame_ransac = T_ransac / 2.0
        for (tt, te, tf, used) in frame_timings:
            total_ms = (tt + te + tf + per_frame_match + per_frame_ransac) * 1000.0
            if used:
                total_triggered_ms.append(total_ms)
            else:
                total_bypassed_ms.append(total_ms)

    # ---------- 平均 ----------
    T_trig   = np.mean(T_trig_list)
    T_enh    = np.mean(T_enh_list)
    T_feat   = np.mean(T_feat_list)
    T_match  = np.mean(T_match_list)  / 2.0   # per-frame
    T_ransac = np.mean(T_ransac_list) / 2.0   # per-frame
    T_total  = T_trig + T_enh + T_feat + T_match + T_ransac

    trig_rate = trig_hits / max(trig_total, 1)

    print(f"\n===== {method_name} =====")
    print(f"Trigger     : {T_trig*1000:7.2f} ms   "
          f"(activation rate = {trig_rate*100:.1f}%)")
    print(f"Enhancement : {T_enh*1000:7.2f} ms   "
          f"(avg across triggered+bypassed frames)")
    print(f"Feature     : {T_feat*1000:7.2f} ms")
    print(f"Matching    : {T_match*1000:7.2f} ms")
    print(f"RANSAC      : {T_ransac*1000:7.2f} ms")
    print("--------------------------------")
    print(f"Total (avg) : {T_total*1000:7.2f} ms")
    print(f"FPS  (avg)  : {1.0/T_total:7.2f}")

    if total_triggered_ms:
        m = np.mean(total_triggered_ms)
        print(f"  → Triggered path : {m:7.2f} ms  |  {1000.0/m:.2f} FPS   "
              f"(n={len(total_triggered_ms)})")
    if total_bypassed_ms:
        m = np.mean(total_bypassed_ms)
        print(f"  → Bypassed  path : {m:7.2f} ms  |  {1000.0/m:.2f} FPS   "
              f"(n={len(total_bypassed_ms)})")

    return T_total


# ===============================
# 主程序
# ===============================
if __name__ == "__main__":

    model_path = 'lolv1.tflite'
    # case 1: normal frames
    # input_image_path1 = 'datasets\\RealData\\video2_walk\\frame_0040.png'
    # input_image_path2 = 'datasets\\RealData\\video2_walk\\frame_0041.png'
    
    # case 2: bright flash moment(only for bypass path testing)
    input_image_path1 = 'datasets\\RealData\\video_flash_moment\\frame_0052.png'
    input_image_path2 = 'datasets\\RealData\\video_flash_moment\\frame_0052.png'
    
    interpreter = lite.Interpreter(model_path=model_path, num_threads=4)
    interpreter.allocate_tensors()

    # 读取两张连续帧
    img1 = cv2.imread(input_image_path1)
    img1 = cv2.resize(img1, (600, 400))
    img2 = cv2.imread(input_image_path2)
    img2 = cv2.resize(img2, (600, 400))

    # ===============================
    # 预热（非常重要）
    # ===============================
    print("Warming up...")

    dummy = np.random.randint(0, 255, (400, 600, 3), dtype=np.uint8)
    dummy_gray = cv2.cvtColor(dummy, cv2.COLOR_BGR2GRAY)

    for _ in range(10):
        enhance_image_cv(dummy, interpreter)

    for _ in range(10):
        orb.detectAndCompute(dummy_gray, None)

    # warm up the trigger path too
    for _ in range(20):
        _ = cv2.mean(dummy_gray)[0]

    print("Warm-up done.")

    # ===============================
    # 各方法测试
    # ===============================

    run_pipeline(img1, img2, "Original", enhance_func=None)

    run_pipeline(img1, img2, "MobileIE (every frame)",
                 enhance_func=lambda x: enhance_image_cv(x, interpreter))

    run_pipeline(img1, img2, "Gamma",
                 enhance_func=lambda x: enhance_gamma(x, gamma=4))

    run_pipeline(img1, img2, "HE",
                 enhance_func=enhance_he)

    run_pipeline(img1, img2, "CLAHE",
                 enhance_func=enhance_clahe)

    # ---------- Ours: Adaptive Trigger ----------
    run_pipeline_adaptive(
        img1, img2,
        "Ours (Adaptive Trigger)",
        enhance_func=lambda x: enhance_image_cv(x, interpreter),
        tau_mu=TAU_MU,
        tau_I=TAU_I,
        repeat=3,
    )