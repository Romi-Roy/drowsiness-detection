from scipy.spatial import distance
import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

def eye_aspect_ratio(eye_landmarks):
    # Calculate the EAR using the distance formula for the given eye landmarks
    A = distance.euclidean(eye_landmarks[1], eye_landmarks[5])
    B = distance.euclidean(eye_landmarks[2], eye_landmarks[4])
    C = distance.euclidean(eye_landmarks[0], eye_landmarks[3])
    ear = (A + B) / (2.0 * C)
    return ear

# EAR threshold and frame check settings
thresh = 0.25
frame_check = 20
flag = 0

# Capture from default camera
cap = cv2.VideoCapture(0)  # Adjust to correct camera index or path

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert frame to RGB for Mediapipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Extract landmarks for the eyes
            left_eye_landmarks = [np.array([face_landmarks.landmark[i].x, face_landmarks.landmark[i].y]) 
                                  for i in [33, 160, 158, 133, 153, 144]]  # Example indices for left eye
            right_eye_landmarks = [np.array([face_landmarks.landmark[i].x, face_landmarks.landmark[i].y]) 
                                   for i in [362, 385, 387, 263, 373, 380]]  # Example indices for right eye

            # Compute EAR
            leftEAR = eye_aspect_ratio(left_eye_landmarks)
            rightEAR = eye_aspect_ratio(right_eye_landmarks)
            ear = (leftEAR + rightEAR) / 2.0

            # Draw landmarks on frame
            for landmark in left_eye_landmarks + right_eye_landmarks:
                x, y = int(landmark[0] * frame.shape[1]), int(landmark[1] * frame.shape[0])
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

            # Check for drowsiness
            if ear < thresh:
                flag += 1
                if flag >= frame_check:
                    cv2.putText(frame, "****************ALERT!****************", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                flag = 0

            # Display EAR on the frame for debugging
            cv2.putText(frame, f"EAR: {ear:.2f}", (500, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow('Drowsiness Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
