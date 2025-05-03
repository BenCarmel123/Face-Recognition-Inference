import os
import cv2
import mediapipe as mp
from tqdm import tqdm

# === Mediapipe Setup ===
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

def detect_and_crop_face(frame):
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_frame)

    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            x_min = int(bboxC.xmin * w)
            y_min = int(bboxC.ymin * h)
            x_max = int((bboxC.xmin + bboxC.width) * w)
            y_max = int((bboxC.ymin + bboxC.height) * h)

            # Ensure the bounding box is within image bounds
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(w, x_max)
            y_max = min(h, y_max)

            cropped_face = frame[y_min:y_max, x_min:x_max]
            if cropped_face.size > 0:
                return cropped_face
    return None

def crop_all_images_in_folder(folder_path):
    print(f"🔍 Cropping faces in: {folder_path}")
    for filename in tqdm(os.listdir(folder_path)):
        file_path = os.path.join(folder_path, filename)

        # Check if it's an image
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        try:
            image = cv2.imread(file_path)
            if image is None:
                print(f"⚠️ Skipped unreadable image: {filename}")
                continue

            cropped = detect_and_crop_face(image)

            # If cropping worked, overwrite original
            if cropped is not None:
                cv2.imwrite(file_path, cropped)
            else:
                print(f"⚠️ No face detected in: {filename}, kept original.")
        except Exception as e:
            print(f"❌ Failed to process {filename}: {e}")

    print("✅ Finished cropping all valid faces.")

# === Run it ===
if __name__ == "__main__":
    crop_all_images_in_folder("data/trainCrop")
