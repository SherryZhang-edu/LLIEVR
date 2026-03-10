import cv2
import numpy as np
import os
import matplotlib.pyplot as plt


# -----------------------------
# TUM 内参（Freiburg1）
# -----------------------------
K = np.array([
    [517.3, 0, 318.6],
    [0, 516.5, 255.3],
    [0, 0, 1]
])


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
                image_paths.append(os.path.join(sequence_path, filename))

    return image_paths


def main(sequence_path, max_frames=500):

    image_list = load_tum_rgb_list(sequence_path)

    orb = cv2.ORB_create(nfeatures=1500)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    # 世界坐标初始化
    R_total = np.eye(3)
    t_total = np.zeros((3, 1))

    trajectory = []

    prev_img = cv2.imread(image_list[0])
    prev_gray = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
    prev_kp, prev_des = orb.detectAndCompute(prev_gray, None)

    for i in range(1, min(len(image_list), max_frames)):

        img = cv2.imread(image_list[i])
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        kp, des = orb.detectAndCompute(gray, None)

        if prev_des is None or des is None:
            continue

        matches = bf.knnMatch(prev_des, des, k=2)

        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append(m)

        if len(good) < 8:
            continue

        pts1 = np.float32([prev_kp[m.queryIdx].pt for m in good])
        pts2 = np.float32([kp[m.trainIdx].pt for m in good])

        # Essential Matrix
        E, mask = cv2.findEssentialMat(
            pts1,
            pts2,
            K,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=1.0
        )

        if E is None:
            continue

        _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, K)

        # 累计位姿
        t_total += R_total @ t
        R_total = R @ R_total

        trajectory.append((t_total[0][0], t_total[2][0]))

        print(f"Frame {i}: Inliers={int(mask.sum())}")

        prev_kp = kp
        prev_des = des

    # -----------------------------
    # 轨迹可视化
    # -----------------------------
    trajectory = np.array(trajectory)

    plt.figure(figsize=(6, 6))
    plt.plot(trajectory[:, 0], trajectory[:, 1])
    plt.xlabel("X")
    plt.ylabel("Z")
    plt.title("Estimated Camera Trajectory")
    plt.axis("equal")
    plt.grid()
    plt.show()


if __name__ == "__main__":
    sequence_path = "datasets/lle/rgbd_desk"
    main(sequence_path)