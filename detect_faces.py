import os
import cv2
from facenet_pytorch import MTCNN

# Input frames folder
frames_folder = "frames"

# Output faces folder
faces_folder = "faces"

os.makedirs(faces_folder, exist_ok=True)

# Load MTCNN model
mtcnn = MTCNN()

# Read all frame images
frame_files = os.listdir(frames_folder)

print("Total Frames Found:", len(frame_files))

for frame_name in frame_files:

    frame_path = os.path.join(frames_folder, frame_name)

    # Read image
    frame = cv2.imread(frame_path)

    if frame is None:
        continue

    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect face
    boxes, _ = mtcnn.detect(rgb_frame)

    # If face detected
    if boxes is not None:

        for i, box in enumerate(boxes):

            x1, y1, x2, y2 = box.astype(int)

            # Draw rectangle
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # Crop face
            face = frame[y1:y2, x1:x2]

            # Save cropped face
            face_path = os.path.join(
                faces_folder,
                f"{frame_name}_face_{i}.jpg"
            )

            cv2.imwrite(face_path, face)

    # Show frame
    cv2.imshow("Face Detection", frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

print("Face Detection Completed")               