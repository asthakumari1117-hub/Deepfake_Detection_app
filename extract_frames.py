import cv2
import os

# Video path
video_path = "videos/Original/000.mp4"

# Create output folder
output_folder = "frames"

os.makedirs(output_folder, exist_ok=True)

# Open video
cap = cv2.VideoCapture(video_path)

# Check video opened
if not cap.isOpened():
    print("Error Opening Video")
    exit()

# Video info
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print("FPS:", fps)
print("Total Frames:", total_frames)

frame_count = 0

frame_skip = 30

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Save only every 5th frame
    if frame_count % frame_skip == 0:

        frame_path = os.path.join(
            output_folder,
            f"frame_{frame_count}.jpg"
        )

        cv2.imwrite(frame_path, frame)

        print(f"Saved: {frame_path}")

    frame_count += 1

cap.release()

print("Frame Extraction Completed")