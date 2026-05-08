import os
import cv2
import numpy as np

# Input folder
faces_folder = "faces"

# Output folder
output_folder = "output"

os.makedirs(output_folder, exist_ok=True)

# Read all face images
face_files = os.listdir(faces_folder)

print("Total Faces Found:", len(face_files))

for face_name in face_files:

    face_path = os.path.join(faces_folder, face_name)

    # Read image
    face = cv2.imread(face_path)

    if face is None:
        continue

    # Resize to 224x224
    resized_face = cv2.resize(face, (224, 224))

    # Normalize pixels
    normalized_face = resized_face / 255.0

    # Convert back for saving
    save_image = (normalized_face * 255).astype(np.uint8)

    # Save processed image
    save_path = os.path.join(output_folder, face_name)

    cv2.imwrite(save_path, save_image)

    print(f"Processed: {face_name}")

print("Preprocessing Completed")