import os
import cv2
from facenet_pytorch import MTCNN

# =========================
# INPUT FRAME FOLDERS
# =========================

real_frames_folder = "processed_dataset/real"
fake_frames_folder = "processed_dataset/fake"

# =========================
# OUTPUT FACE FOLDERS
# =========================

real_faces_folder = "processed_faces/real"
fake_faces_folder = "processed_faces/fake"

os.makedirs(real_faces_folder, exist_ok=True)
os.makedirs(fake_faces_folder, exist_ok=True)

# =========================
# LOAD MTCNN MODEL
# =========================

mtcnn = MTCNN(
    keep_all=True
)

# =========================
# FACE DETECTION FUNCTION
# =========================

def detect_faces(input_folder, output_folder, label):

    frame_files = os.listdir(input_folder)

    print(f"\nTotal {label} Frames Found:", len(frame_files))

    for frame_name in frame_files:

        frame_path = os.path.join(input_folder, frame_name)

        # Read image
        frame = cv2.imread(frame_path)

        if frame is None:
            continue

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces
        boxes, probs = mtcnn.detect(rgb_frame)

        # If face detected
        if boxes is not None:

            for i, box in enumerate(boxes):

                # =========================
                # CONFIDENCE FILTER
                # =========================

                confidence = probs[i]

                # Ignore weak detections
                if confidence < 0.98:
                    continue

                x1, y1, x2, y2 = box.astype(int)

                # =========================
                # SIZE FILTER
                # =========================

                width = x2 - x1
                height = y2 - y1

                # Ignore tiny detections
                if width < 100 or height < 100:
                    continue

                # =========================
                # ASPECT RATIO FILTER
                # =========================

                ratio = width / height

                # Ignore weird shapes
                if ratio < 0.6 or ratio > 1.5:
                    continue

                # Prevent negative coordinates
                x1 = max(0, x1)
                y1 = max(0, y1)

                # Crop face
                face = frame[y1:y2, x1:x2]

                if face.size == 0:
                    continue

                # Draw rectangle
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                # Save face
                face_path = os.path.join(
                    output_folder,
                    f"{frame_name}_face_{i}.jpg"
                )

                cv2.imwrite(face_path, face)

                print(f"{label} Face Saved:", face_path)

        # Show detection result
        cv2.imshow(f"{label} Face Detection", frame)

        # Press Q to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

# =========================
# PROCESS REAL FRAMES
# =========================

detect_faces(
    real_frames_folder,
    real_faces_folder,
    "REAL"
)

# =========================
# PROCESS FAKE FRAMES
# =========================

detect_faces(
    fake_frames_folder,
    fake_faces_folder,
    "FAKE"
)

print("\nALL FACE DETECTION COMPLETED")