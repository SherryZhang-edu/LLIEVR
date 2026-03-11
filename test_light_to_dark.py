from orb_ransac_pipeline_visual import match_and_visualize

img1_pth = "datasets\\RealData\\video3_flash\\frame_0024.png"
# img2_pth = "datasets\\RealData\\video3_flash\\frame_0022.png"
img2_pth = "datasets\\RealData\\video3_flash\\frame_0025.png"
img3_pth = "experiments\\results\\RealData\\video3_flash_enhanced\\frame_0024.png"
img4_pth = "experiments\\results\\RealData\\video3_flash_enhanced\\frame_0027.png"

import cv2
img1 = cv2.imread(img1_pth)
img2 = cv2.imread(img2_pth)
img3 = cv2.imread(img3_pth)
img4 = cv2.imread(img4_pth)

orb = cv2.ORB_create(nfeatures=1500)
bf = cv2.BFMatcher(cv2.NORM_HAMMING)

vis, kp, matches, inliers = match_and_visualize(
    img1, img2, orb, bf
)

vis_enhanced, kp_enhanced, matches_enhanced, inliers_enhanced = match_and_visualize(
    img1, img4, orb, bf
)

print(f"RAW: KP={kp}, Matches={matches}, Inliers={inliers}")
print(f"ENHANCED: KP={kp_enhanced}, Matches={matches_enhanced}, Inliers={inliers_enhanced}")

# plot side by side
# import matplotlib.pyplot as plt
# plt.figure(figsize=(12, 6))
# plt.subplot(1, 2, 1)
# plt.title("RAW")
# plt.imshow(vis)
# plt.axis('off')
# plt.subplot(1, 2, 2)
# plt.title("ENHANCED")
# plt.imshow(vis_enhanced)
# plt.axis('off')
# plt.tight_layout()  
# plt.show()