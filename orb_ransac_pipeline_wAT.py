import cv2
import numpy as np
from utils.evaluation import compute_tentative_matches, compute_ransac_inliers, compute_inlier_ratio, compute_tracking_failure_rate

def evaluate_sequence(
    image_list,
    enhance_fn=None,
    inlier_threshold=30
):

    orb = cv2.ORB_create(nfeatures=1500)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    keypoints_list = []
    tentative_matches_list = []
    inliers_list = []
    inlier_ratio_list = []

    prev_img = cv2.imread(image_list[0])
    if enhance_fn is not None:
        prev_img = enhance_fn(prev_img)

    prev_gray = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
    prev_kp, prev_des = orb.detectAndCompute(prev_gray, None)

    for i in range(1, len(image_list)):

        img = cv2.imread(image_list[i])

        if enhance_fn is not None:
            img = enhance_fn(img)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, des = orb.detectAndCompute(gray, None)

        # 1️⃣ keypoints
        keypoints_list.append(len(kp))

        if prev_des is None or des is None:
            prev_kp, prev_des = kp, des
            continue

        # 2️⃣ tentative matches
        good_matches = compute_tentative_matches(prev_des, des, bf)
        tentative_matches = len(good_matches)
        tentative_matches_list.append(tentative_matches)

        # 3️⃣ ransac inliers
        inliers, mask = compute_ransac_inliers(prev_kp, kp, good_matches)
        inliers_list.append(inliers)

        # 4️⃣ inlier ratio
        ratio = compute_inlier_ratio(inliers, tentative_matches)
        inlier_ratio_list.append(ratio)

        prev_kp, prev_des = kp, des

    # 5️⃣ tracking failure rate
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


if __name__ == "__main__":
    # 这里可以放一些简单的测试代码，或者直接在 test_proxy_on_tum.py 中调用 evaluate_sequence 来测试
    pass