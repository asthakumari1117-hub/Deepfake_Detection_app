import cv2
import os
from facenet_pytorch import MTCNN

# Device
device = 'cpu'

# Face Detector
mtcnn = MTCNN(keep_all=True, device=device)

# Input folders
real_input = "processed_dataset/real"
fake_input = "processed_dataset/fake"

# Output folders
real_output = "processed_faces/real"
fake_output = "processed_faces/fake"

# Create folders
os.makedirs(real_output, exist_ok=True)
os.makedirs(fake_output, exist_ok=True)

# Function
def process_folder(input_folder, output_folder, label):

    image_files = os.listdir(input_folder)

    print(f"\nProcessing {label} Images:")
    print(f"Total Images: {len(image_files)}")

    for image_name in image_files:

        image_path = os.path.join(input_folder, image_name)

        image = cv2.imread(image_path)

        if image is None:
            continue

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        boxes, _ = mtcnn.detect(rgb_image)

        if boxes is not None:

            for i, box in enumerate(boxes):

                x1, y1, x2, y2 = map(int, box)

                face = image[y1:y2, x1:x2]

                if face.size == 0:
                    continue

                save_path = os.path.join(
                    output_folder,
                    f"{image_name}_face_{i}.jpg"
                )

                cv2.imwrite(save_path, face)

                print(f"{label} Face Saved: {save_path}")

# Process real
process_folder(real_input, real_output, "REAL")

# Process fake
process_folder(fake_input, fake_output, "FAKE")

print("\nFull Face Detection Completed")