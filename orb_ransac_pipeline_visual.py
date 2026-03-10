import cv2
import numpy as np
import os
import time
from utils.evaluation import load_darkzurich_rgb_list,load_tum_rgb_list



def match_and_visualize(img1, img2, orb, bf):

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    if des1 is None or des2 is None:
        return None, 0, 0, 0

    matches = bf.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    if len(good) < 8:
        return None, len(kp1), len(good), 0

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])

    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 3.0)

    if mask is None:
        return None, len(kp1), len(good), 0

    inlier_matches = []
    for i, m in enumerate(good):
        if mask[i]:
            inlier_matches.append(m)

    # -------- Visualization --------
    vis = cv2.drawMatches(
        img1, kp1,
        img2, kp2,
        inlier_matches,
        None,
        matchColor=(0, 255, 0),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    # 画投影框（运动可视化）
    h, w = img1.shape[:2]
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)

    projected = cv2.perspectiveTransform(corners, H)

    img2_with_box = img2.copy()
    projected = projected.astype(int)

    for i in range(4):
        pt1 = tuple(projected[i][0])
        pt2 = tuple(projected[(i + 1) % 4][0])
        cv2.line(img2_with_box, pt1, pt2, (0, 0, 255), 2)

    return vis, len(kp1), len(good), len(inlier_matches)


def evaluate_sequence(image_list, save_dir="vis_output", max_frames=50):

    os.makedirs(save_dir, exist_ok=True)

    orb = cv2.ORB_create(nfeatures=1000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    for i in range(min(max_frames, len(image_list) - 1)):

        img1 = cv2.imread(image_list[i])
        img2 = cv2.imread(image_list[i + 1])

        start = time.time()

        vis, kp, matches, inliers = match_and_visualize(
            img1, img2, orb, bf
        )

        elapsed = time.time() - start

        if vis is None:
            continue

        ratio = inliers / matches if matches > 0 else 0

        cv2.putText(vis,
                    f"KP: {kp}",
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2)

        cv2.putText(vis,
                    f"Matches: {matches}",
                    (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2)

        cv2.putText(vis,
                    f"Inliers: {inliers}",
                    (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2)

        cv2.putText(vis,
                    f"Ratio: {ratio:.3f}",
                    (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2)

        cv2.putText(vis,
                    f"Time: {elapsed*1000:.1f} ms",
                    (20, 150),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2)

        save_path = os.path.join(save_dir, f"frame_{i:04d}.png")
        cv2.imwrite(save_path, vis)

        print(f"Saved {save_path}")


if __name__ == "__main__":

    # sequence_path = "datasets/lle/darkZurich/set1_resized"
    sequence_path = "experiments/results/darkZurich/set1_enhanced"  
    image_list = load_darkzurich_rgb_list(sequence_path)
    evaluate_sequence(image_list, save_dir='vis_output/set1_enhanced')