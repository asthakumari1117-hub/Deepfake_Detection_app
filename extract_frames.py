import os
import cv2

# =========================
# REAL VIDEO FOLDER
# =========================

real_videos_folder = "videos/original"

# =========================
# FAKE VIDEO FOLDERS
# =========================

fake_folders = [
    "videos/DeepFakeDetection",
    "videos/Face2Face",
    "videos/FaceSwap",
    "videos/FaceShifter",
    "videos/NeuralTextures"
]

# =========================
# OUTPUT FRAME FOLDERS
# =========================

real_output_folder = "processed_dataset/real"
fake_output_folder = "processed_dataset/fake"

os.makedirs(real_output_folder, exist_ok=True)
os.makedirs(fake_output_folder, exist_ok=True)

# =========================
# FRAME SKIP
# =========================

frame_skip = 5

# =========================
# FRAME EXTRACTION FUNCTION
# =========================

def extract_frames(video_folder, output_folder, label):

    video_files = os.listdir(video_folder)

    print(f"\nTotal {label} Videos in {video_folder}: {len(video_files)}")

    for video_name in video_files:

        video_path = os.path.join(video_folder, video_name)

        # Skip non-video files
        if not video_name.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
            continue

        cap = cv2.VideoCapture(video_path)

        frame_count = 0
        saved_count = 0

        print(f"\nProcessing {label} Video: {video_name}")

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            # Save every nth frame
            if frame_count % frame_skip == 0:

                frame_filename = (
                    f"{video_name.split('.')[0]}"
                    f"_frame_{frame_count}.jpg"
                )

                frame_path = os.path.join(
                    output_folder,
                    frame_filename
                )

                cv2.imwrite(frame_path, frame)

                saved_count += 1

            frame_count += 1

        cap.release()

        print(f"{label} Saved Frames: {saved_count}")

# =========================
# PROCESS REAL VIDEOS
# =========================

extract_frames(
    real_videos_folder,
    real_output_folder,
    "REAL"
)

# =========================
# PROCESS ALL FAKE VIDEOS
# =========================

for fake_folder in fake_folders:

    extract_frames(
        fake_folder,
        fake_output_folder,
        "FAKE"
    )

print("\nALL FRAME EXTRACTION COMPLETED")