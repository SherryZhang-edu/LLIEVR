from orb_ransac_pipeline_visual import match_and_visualize
import cv2
import os

# img1 = cv2.imread("experiments/results/RealData/video2_walk_enhanced/frame_0009.png")
# img2 = cv2.imread("experiments/results/RealData/video2_walk_enhanced/frame_0015.png")
# img1 = cv2.imread("datasets/RealData/video2_walk/frame_0009.png")
# img2 = cv2.imread("datasets/RealData/video2_walk/frame_0015.png")
# save_dir = "results/figures/video_walk"
# img1 = cv2.imread("datasets/RealData/video4_office/frame_0009.png")
# img2 = cv2.imread("datasets/RealData/video4_office/frame_0010.png")
img1 = cv2.imread("experiments/results/RealData/video4_office_enhanced/frame_0009.png")
img2 = cv2.imread("experiments/results/RealData/video4_office_enhanced/frame_0015.png")
save_dir = "results/figures/video_office"

orb = cv2.ORB_create(nfeatures=2000)
bf = cv2.BFMatcher(cv2.NORM_HAMMING)

vis, kp, matches, inliers = match_and_visualize(
    img1, img2, orb, bf
)

save_path = os.path.join(save_dir, "matches_origin.png")
cv2.imwrite(save_path, vis)
print(f"Saved {save_path}")
ratio = inliers / matches if matches > 0 else 0
print(f"KP: {kp}, Matches: {matches}, Inliers: {inliers}, Ratio: {ratio:.3f}")  