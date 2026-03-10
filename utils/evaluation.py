import cv2
import numpy as np
import os

# =========================================================
# 1️⃣ 检测特征点数量
# =========================================================
def compute_detected_keypoints(img, orb):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kp = orb.detect(gray, None)
    return len(kp)


# =========================================================
# 2️⃣ 计算 tentative matches
# =========================================================
def compute_tentative_matches(des1, des2, bf, ratio=0.75):
    matches = bf.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < ratio * n.distance:
            good.append(m)

    return good


# =========================================================
# 3️⃣ 计算 RANSAC inliers（使用 Fundamental Matrix）
# =========================================================
def compute_ransac_inliers(kp1, kp2, matches):

    if len(matches) < 8:
        return 0, None

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    F, mask = cv2.findFundamentalMat(
        pts1,
        pts2,
        cv2.RANSAC,
        ransacReprojThreshold=1.0,
        confidence=0.99
    )

    if mask is None:
        return 0, None

    inliers = int(mask.sum())

    return inliers, mask


# =========================================================
# 4️⃣ Inlier Ratio
# =========================================================
def compute_inlier_ratio(inliers, tentative_matches):

    if tentative_matches == 0:
        return 0.0

    return inliers / tentative_matches


# =========================================================
# 5️⃣ Tracking Failure Rate
# =========================================================
def compute_tracking_failure_rate(inlier_list, threshold=30):
    """
    threshold: 低于多少 inliers 视为 tracking failure
    """
    failures = [1 for i in inlier_list if i < threshold]
    return sum(failures) / len(inlier_list)



# =====================================================
# 读取 TUM rgb.txt
# =====================================================

def load_tum_rgb_list(sequence_path):

    rgb_txt = os.path.join(sequence_path, "rgb.txt")
    image_paths = []

    with open(rgb_txt, "r") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split()
            if len(parts) == 2:
                _, filename = parts
                full_path = os.path.join(sequence_path, filename)
                image_paths.append(full_path)

    return image_paths

def load_darkzurich_rgb_list(directory_path):
    """
    加载指定文件夹下的所有图片路径，并按文件名（时间/帧号）排序。
    :param directory_path: 包含图片的文件夹路径
    :return: 排好序的图片完整路径列表 (list)
    """
    # 定义常见的图片扩展名
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    
    image_paths = []
    
    # 检查文件夹是否存在
    if not os.path.exists(directory_path):
        print(f"警告: 文件夹 '{directory_path}' 不存在！")
        return image_paths
        
    # 遍历文件夹中的所有文件
    for filename in os.listdir(directory_path):
        # 检查后缀名，确保只读取图片文件（忽略大小写）
        if filename.lower().endswith(valid_extensions):
            # 拼接完整路径
            full_path = os.path.join(directory_path, filename)
            image_paths.append(full_path)
            
    # 对路径列表进行原地排序
    # 因为文件名中的帧号是补零的（如 000294），默认的字母顺序排序就能保证时间顺序正确
    image_paths.sort()
    
    return image_paths