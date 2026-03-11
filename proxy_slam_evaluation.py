import cv2
import numpy as np
from utils.enhancer import enhance_image_cv,load_model

prev_inliers = 1000

# =========================================================
# 检测特征点数量
# =========================================================
def compute_detected_keypoints(img, orb):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kp = orb.detect(gray, None)
    return len(kp)


# =========================================================
# 计算 tentative matches
# =========================================================
def compute_tentative_matches(des1, des2, bf, ratio=0.75):
    matches = bf.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < ratio * n.distance:
            good.append(m)

    return good


# =========================================================
# 计算 RANSAC inliers（使用 Fundamental Matrix）
# =========================================================
def compute_ransac_inliers(kp1, kp2, matches):

    if len(matches) < 8:
        return 0, None

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    H, mask = cv2.findHomography(
        pts1,
        pts2,
        cv2.RANSAC,
        3.0
    )

    if mask is None:
        return 0, None

    inliers = int(mask.sum())

    return inliers, mask


# =========================================================
# Inlier Ratio
# =========================================================
def compute_inlier_ratio(inliers, tentative_matches):

    if tentative_matches == 0:
        return 0.0

    return inliers / tentative_matches


# =========================================================
# Tracking Failure Rate
# =========================================================
def compute_tracking_failure_rate(inlier_list, threshold=30):
    """
    threshold: 低于多少 inliers 视为 tracking failure
    """
    failures = [1 for i in inlier_list if i < threshold]
    return sum(failures) / len(inlier_list)




# =========================================================
# 主评估函数
# =========================================================
def evaluate_sequence(
    image_list,
    enhance = False,
    enhance_mode='every',
    inlier_threshold=30,
    model_path='lolv1.tflite',
    target_size=(600, 400),
    tau_mu=60,
    tau_I=30
):
    
    interpreter = load_model(model_path)
    
    orb = cv2.ORB_create(nfeatures=1500)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    keypoints_list = []
    tentative_matches_list = []
    inliers_list = []
    inlier_ratio_list = []

    prev_img = cv2.imread(image_list[0])
    # resize image if it is not the target size
    if (prev_img.shape[1], prev_img.shape[0]) != target_size:
        prev_img = cv2.resize(prev_img, target_size, interpolation=cv2.INTER_CUBIC)
    
    
    if enhance:
        prev_img = enhance_image_cv(prev_img, interpreter)

    prev_gray = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
    prev_kp, prev_des = orb.detectAndCompute(prev_gray, None)
    
    prev_inliers = 100
    for i in range(1, len(image_list)):

        img = cv2.imread(image_list[i])
        # resize image if it is not the target size
        if (img.shape[1], img.shape[0]) != target_size:
            img = cv2.resize(img, target_size, interpolation=cv2.INTER_CUBIC)


        # ===============================
        # Adaptive trigger
        # ===============================
        if enhance: 
            if enhance_mode == 'every':
                img = enhance_image_cv(img, interpreter)
            elif enhance_mode == 'adaptive':
                # 计算亮度
                gray_temp = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                mu = np.mean(gray_temp)
                # g_t = (mu < tau_mu) or (prev_inliers < tau_I) # tau_I 不必要
                g_t = (mu < tau_mu)   
                if g_t:
                    img = enhance_image_cv(img, interpreter)
                    print(f"File {image_list[i]}: Enhanced (mu={mu:.1f}, prev_inliers={prev_inliers})")
        else:
            pass  # 不增强，直接使用原图

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, des = orb.detectAndCompute(gray, None)

        # keypoints
        keypoints_list.append(len(kp))

        if prev_des is None or des is None:
            prev_kp, prev_des = kp, des
            continue

        # tentative matches
        good_matches = compute_tentative_matches(prev_des, des, bf)
        tentative_matches = len(good_matches)
        tentative_matches_list.append(tentative_matches)

        # ransac inliers
        inliers, mask = compute_ransac_inliers(prev_kp, kp, good_matches)
        inliers_list.append(inliers)

        # inlier ratio
        ratio = compute_inlier_ratio(inliers, tentative_matches)
        inlier_ratio_list.append(ratio)

        prev_kp, prev_des = kp, des
        prev_inliers = inliers

    # tracking failure rate
    failure_rate = compute_tracking_failure_rate(
        inliers_list,
        threshold=inlier_threshold
    )

    # ===========================
    # 汇总统计
    # ===========================
    results = {
        "avg_keypoints": np.mean(keypoints_list) if keypoints_list else 0,
        "avg_tentative_matches": np.mean(tentative_matches_list) if tentative_matches_list else 0,
        "avg_inliers": np.mean(inliers_list) if inliers_list else 0,
        "avg_inlier_ratio": np.mean(inlier_ratio_list) if inlier_ratio_list else 0,
        "tracking_failure_rate": failure_rate
    }

    return results