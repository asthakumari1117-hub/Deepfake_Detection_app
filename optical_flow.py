import os
import cv2
import numpy as np

# -----------------------------
# PROCESS FUNCTION
# -----------------------------

def process_folder(folder_path, label):

    print("\n======================")
    print("PROCESSING:", label)
    print("======================")

    # Read image files only
    frame_files = []

    for file in os.listdir(folder_path):

        if file.endswith(".jpg") or file.endswith(".png"):

            frame_files.append(file)

    frame_files = sorted(frame_files)

    print("Total Frames:", len(frame_files))

    # Empty check
    if len(frame_files) == 0:

        print("No image frames found")
        return

    # First frame
    first_frame_path = os.path.join(
        folder_path,
        frame_files[0]
    )

    prev_frame = cv2.imread(first_frame_path)

    if prev_frame is None:

        print("First frame corrupted")
        return

    # Resize
    prev_frame = cv2.resize(
        prev_frame,
        (224, 224)
    )

    # Gray
    prev_gray = cv2.cvtColor(
        prev_frame,
        cv2.COLOR_BGR2GRAY
    )

    # -----------------------------
    # LOOP THROUGH FRAMES
    # -----------------------------

    for i in range(1, len(frame_files)):

        frame_path = os.path.join(
            folder_path,
            frame_files[i]
        )

        frame = cv2.imread(frame_path)

        if frame is None:

            print(f"Skipped Frame: {i}")
            continue

        # Resize
        frame = cv2.resize(
            frame,
            (224, 224)
        )

        # Gray
        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        # Optical Flow
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray,
            gray,
            None,
            0.5,
            3,
            15,
            3,
            5,
            1.2,
            0
        )

        # Magnitude
        magnitude, angle = cv2.cartToPolar(
            flow[..., 0],
            flow[..., 1]
        )

        magnitude = cv2.normalize(
            magnitude,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        )

        magnitude = magnitude.astype(np.uint8)

        # Show
        cv2.imshow(
            f"Optical Flow - {label}",
            magnitude
        )

        print(f"{label} Processed Frame: {i}")

        # Update
        prev_gray = gray

        # Quit
        if cv2.waitKey(30) & 0xFF == ord('q'):

            break

    cv2.destroyAllWindows()

    print(f"\n{label} Optical Flow Completed")

# -----------------------------
# RUN BOTH
# -----------------------------

real_folder = "processed_dataset/real"

fake_folder = "processed_dataset/fake"

process_folder(real_folder, "REAL")

process_folder(fake_folder, "FAKE")

print("\nALL OPTICAL FLOW PROCESSING COMPLETED")