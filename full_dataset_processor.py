import cv2
import os

# -----------------------------
# DATASET PATH
# -----------------------------

dataset_path = "videos"

# -----------------------------
# OUTPUT PATHS
# -----------------------------

real_output = "processed_dataset/real"
fake_output = "processed_dataset/fake"

os.makedirs(real_output, exist_ok=True)
os.makedirs(fake_output, exist_ok=True)

# -----------------------------
# FAKE FOLDERS
# -----------------------------

fake_folders = [
    "DeepFakeDetection",
    "Deepfakes",
    "Face2Face",
    "FaceShifter",
    "FaceSwap",
    "NeuralTextures"
]

# -----------------------------
# PROCESS REAL VIDEOS
# -----------------------------

real_folder = os.path.join(
    dataset_path,
    "original"
)

print("\nProcessing REAL videos...\n")

for video_name in os.listdir(real_folder):

    video_path = os.path.join(
        real_folder,
        video_name
    )

    cap = cv2.VideoCapture(video_path)

    frame_count = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        # Every 10th frame
        if frame_count % 30 == 0:

            save_name = (
                f"REAL_{video_name[:-4]}"
                f"_frame_{frame_count}.jpg"
            )

            save_path = os.path.join(
                real_output,
                save_name
            )

            cv2.imwrite(save_path, frame)

            print(f"REAL Saved: {save_name}")

        frame_count += 1

    cap.release()

# -----------------------------
# PROCESS FAKE VIDEOS
# -----------------------------

print("\nProcessing FAKE videos...\n")

for folder in fake_folders:

    folder_path = os.path.join(
        dataset_path,
        folder
    )

    print(f"\nFolder: {folder}")

    for video_name in os.listdir(folder_path):

        video_path = os.path.join(
            folder_path,
            video_name
        )

        cap = cv2.VideoCapture(video_path)

        frame_count = 0

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            if frame_count % 30 == 0:

                save_name = (
                    f"{folder}_{video_name[:-4]}"
                    f"_frame_{frame_count}.jpg"
                )

                save_path = os.path.join(
                    fake_output,
                    save_name
                )

                cv2.imwrite(save_path, frame)

                print(f"FAKE Saved: {save_name}")

            frame_count += 1

        cap.release()

print("\nFull Dataset Processing Completed")