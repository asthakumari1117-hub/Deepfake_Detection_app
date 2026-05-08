import os
import cv2
import numpy as np

# -----------------------------
# INPUT FOLDERS
# -----------------------------

real_input = "processed_faces/real"
fake_input = "processed_faces/fake"

# -----------------------------
# OUTPUT FOLDERS
# -----------------------------

real_output = "aligned_faces/real"
fake_output = "aligned_faces/fake"

os.makedirs(real_output, exist_ok=True)
os.makedirs(fake_output, exist_ok=True)

# -----------------------------
# FACE DETECTOR
# -----------------------------

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

# -----------------------------
# EYE DETECTOR
# -----------------------------

eye_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_eye.xml"
)

# -----------------------------
# ALIGN FUNCTION
# -----------------------------

def align_face(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    if len(faces) == 0:

        return image

    for (x, y, w, h) in faces:

        face = image[y:y+h, x:x+w]

        face_gray = gray[y:y+h, x:x+w]

        eyes = eye_detector.detectMultiScale(
            face_gray
        )

        # Need minimum 2 eyes
        if len(eyes) >= 2:

            eye_1 = eyes[0]
            eye_2 = eyes[1]

            # Eye centers
            x1 = int(eye_1[0] + eye_1[2] / 2)
            y1 = int(eye_1[1] + eye_1[3] / 2)

            x2 = int(eye_2[0] + eye_2[2] / 2)
            y2 = int(eye_2[1] + eye_2[3] / 2)

            # -----------------------------
            # DRAW GREEN LANDMARKS
            # -----------------------------

            cv2.circle(
                face,
                (x1, y1),
                5,
                (0, 255, 0),
                -1
            )

            cv2.circle(
                face,
                (x2, y2),
                5,
                (0, 255, 0),
                -1
            )

            # Draw eye line
            cv2.line(
                face,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # -----------------------------
            # CALCULATE ANGLE
            # -----------------------------

            angle = np.degrees(
                np.arctan2(
                    y2 - y1,
                    x2 - x1
                )
            )

            center = (
                int(w / 2),
                int(h / 2)
            )

            matrix = cv2.getRotationMatrix2D(
                center,
                float(angle),
                1.0
            )

            aligned = cv2.warpAffine(
                face,
                matrix,
                (w, h)
            )

            aligned = cv2.resize(
                aligned,
                (224, 224)
            )

            return aligned

        # If eyes not found
        face = cv2.resize(
            face,
            (224, 224)
        )

        return face

    return image

# -----------------------------
# PROCESS FUNCTION
# -----------------------------

def process_folder(
    input_folder,
    output_folder,
    label
):

    print(f"\nPROCESSING {label}")

    files = os.listdir(input_folder)

    for file in files:

        if not (
            file.endswith(".jpg")
            or file.endswith(".png")
        ):

            continue

        path = os.path.join(
            input_folder,
            file
        )

        image = cv2.imread(path)

        if image is None:

            print("Skipped:", file)
            continue

        aligned_face = align_face(image)

        save_path = os.path.join(
            output_folder,
            file
        )

        cv2.imwrite(
            save_path,
            aligned_face
        )

        print(f"{label} Saved:", file)

# -----------------------------
# RUN BOTH
# -----------------------------

process_folder(
    real_input,
    real_output,
    "REAL"
)

process_folder(
    fake_input,
    fake_output,
    "FAKE"
)

print("\nFACE ALIGNMENT COMPLETED SUCCESSFULLY")