import cv2
import numpy as np
import os
import time
from utils.evaluation import load_darkzurich_rgb_list,load_tum_rgb_list



def match_orb_ransac(img1, img2, orb, bf):

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    if des1 is None or des2 is None:
        return 0, 0, 0

    matches = bf.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    if len(good) < 8:
        return len(kp1), len(good), 0

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])

    H, mask = cv2.findHomography(
        pts1,
        pts2,
        cv2.RANSAC,
        3.0
    )

    if mask is None:
        return len(kp1), len(good), 0

    inliers = int(mask.sum())

    return len(kp1), len(good), inliers



def evaluate_sequence(image_list, max_frames=200):

    print("Total images:", len(image_list))

    orb = cv2.ORB_create(nfeatures=1000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    total_keypoints = 0
    total_matches = 0
    total_inliers = 0
    total_time = 0

    frame_count = min(len(image_list) - 1, max_frames)

    for i in range(frame_count):

        img1 = cv2.imread(image_list[i])
        img2 = cv2.imread(image_list[i + 1])

        start = time.time()

        kp, matches, inliers = match_orb_ransac(img1, img2, orb, bf)

        elapsed = time.time() - start

        total_keypoints += kp
        total_matches += matches
        total_inliers += inliers
        total_time += elapsed

        if matches > 0:
            inlier_ratio = inliers / matches
        else:
            inlier_ratio = 0

        print(f"Frame {i}: KP={kp}, Matches={matches}, "
              f"Inliers={inliers}, Ratio={inlier_ratio:.3f}, "
              f"Time={elapsed*1000:.2f}ms")

    print("\n===== Summary =====")
    print("Avg keypoints:", total_keypoints / frame_count)
    print("Avg matches:", total_matches / frame_count)
    print("Avg inliers:", total_inliers / frame_count)

    if total_matches > 0:
        print("Overall inlier ratio:", total_inliers / total_matches)

    print("Avg time per frame:", (total_time / frame_count) * 1000, "ms")
    print("FPS:", frame_count / total_time)



if __name__ == "__main__":

    # sequence_path = "datasets/lle/rgbd_desk"
    # image_list = load_tum_rgb_list(sequence_path)
    sequence_path = "datasets/lle/darkZurich/set1_resized"  
    image_list = load_darkzurich_rgb_list(sequence_path)
    evaluate_sequence(image_list, max_frames=200)