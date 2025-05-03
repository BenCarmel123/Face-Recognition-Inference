import cv2
import mediapipe as mp

# Initialize Mediapipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

def detect_and_crop_face(frame):
    h, w, _ = frame.shape
    # Convert frame to RGB for Mediapipe processing - check if needed
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_frame)

    if results.detections:
        for detection in results.detections:
            # Extract bounding box coordinates
            bboxC = detection.location_data.relative_bounding_box
            x_min = int(bboxC.xmin * w)
            y_min = int(bboxC.ymin * h)
            x_max = int((bboxC.xmin + bboxC.width) * w)
            y_max = int((bboxC.ymin + bboxC.height) * h)

            # Crop and display the face
            cropped_face = frame[y_min:y_max, x_min:x_max]

            # Show the cropped face in a separate window
            if cropped_face.size > 0:
                return cropped_face
        
    return None

if __name__ == "__main__":

    # Start webcam
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        cropped_face = detect_and_crop_face(frame)

        # Show the cropped face in a separate window
        if cropped_face is not None:
            cv2.imshow('Cropped Face', cv2.resize(cropped_face, (200, 200)))

        # Exit loop on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()