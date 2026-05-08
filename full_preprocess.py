import cv2
import os

# Input folders
real_input = "processed_faces/real"
fake_input = "processed_faces/fake"

# Output folders
real_output = "final_faces/real"
fake_output = "final_faces/fake"

# Create folders
os.makedirs(real_output, exist_ok=True)
os.makedirs(fake_output, exist_ok=True)

# Image size
IMG_SIZE = 224

# Function
def preprocess_folder(input_folder, output_folder, label):

    image_files = os.listdir(input_folder)

    print(f"\nProcessing {label} Faces")
    print(f"Total Images: {len(image_files)}")

    for image_name in image_files:

        image_path = os.path.join(input_folder, image_name)

        image = cv2.imread(image_path)

        if image is None:
            continue

        # Resize
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

        # Normalize
        image = image / 255.0

        # Convert back for saving
        image = (image * 255).astype("uint8")

        save_path = os.path.join(output_folder, image_name)

        cv2.imwrite(save_path, image)

        print(f"{label} Processed: {save_path}")

# Process real
preprocess_folder(real_input, real_output, "REAL")

# Process fake
preprocess_folder(fake_input, fake_output, "FAKE")

print("\nFull Preprocessing Completed")