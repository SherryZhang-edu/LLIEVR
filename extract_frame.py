import os
import cv2
import csv

# ==============================
# Configuration
# ==============================

VIDEO_DIR = "video/data"          # 输入视频目录
OUTPUT_DIR = "datasets/RealData"        # 输出帧目录
TARGET_FPS = 5                # 抽帧FPS
RESIZE_WIDTH = 600            # 统一宽度
RESIZE_HEIGHT = 400           # 统一高度

# ==============================
# Create output folder
# ==============================

os.makedirs(OUTPUT_DIR, exist_ok=True)

metadata_rows = []

# ==============================
# Process each video
# ==============================

for video_name in os.listdir(VIDEO_DIR):
    if not video_name.lower().endswith((".mp4", ".mov", ".avi")):
        continue

    video_path = os.path.join(VIDEO_DIR, video_name)
    sequence_id = os.path.splitext(video_name)[0]
    sequence_output_dir = os.path.join(OUTPUT_DIR, sequence_id)

    os.makedirs(sequence_output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Failed to open {video_name}")
        continue

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(original_fps / TARGET_FPS)

    print(f"\nProcessing {video_name}")
    print(f"Original FPS: {original_fps}")
    print(f"Frame interval: {frame_interval}")

    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            # Resize
            resized = cv2.resize(frame, (RESIZE_WIDTH, RESIZE_HEIGHT))

            # Save as PNG
            output_name = f"frame_{saved_count:04d}.png"
            output_path = os.path.join(sequence_output_dir, output_name)
            cv2.imwrite(output_path, resized)

            saved_count += 1

        frame_count += 1

    cap.release()

    print(f"Saved {saved_count} frames.")

    # Record metadata
    metadata_rows.append([
        sequence_id,
        video_name,
        original_fps,
        TARGET_FPS,
        RESIZE_WIDTH,
        RESIZE_HEIGHT,
        saved_count
    ])

# ==============================
# Save metadata CSV
# ==============================

metadata_file = os.path.join(OUTPUT_DIR, "metadata.csv")

with open(metadata_file, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "sequence_id",
        "video_file",
        "original_fps",
        "extracted_fps",
        "width",
        "height",
        "num_frames"
    ])
    writer.writerows(metadata_rows)

print("\nAll videos processed successfully.")