from llim_orb_ransac_pipeline import compute_detected_keypoints, compute_tentative_matches, compute_ransac_inliers, compute_inlier_ratio, compute_tracking_failure_rate
import cv2
import numpy as np
from utils.evaluation import load_darkzurich_rgb_list
import os

def evaluate_sequence(image_list,inlier_threshold=30,          
        ):
    orb = cv2.ORB_create(nfeatures=1500)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    
    keypoints_list = []
    tentative_matches_list = []
    inliers_list = []
    inlier_ratio_list = []
    
    prev_img =  cv2.imread(image_list[0])
    
    prev_gray = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
    prev_kp, prev_des = orb.detectAndCompute(prev_gray, None)
    prev_inliers = 100
    for i in range(1, len(image_list)):
        
        img = cv2.imread(image_list[i])
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
    try:
        failure_rate = compute_tracking_failure_rate(
            inliers_list,
            threshold=inlier_threshold
        )
    except:
        failure_rate = 100.0
    
    
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
   
    sequence_path  = "experiments/results/RealData/video3_flash_clahe"
    image_list = load_darkzurich_rgb_list(sequence_path)
    results = evaluate_sequence(image_list)
    print("\n============================")
    print(f"Proxy Evaluation on {os.path.basename(sequence_path)}")
    print("============================")

    print(f"\n{os.path.basename(sequence_path)}:")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")