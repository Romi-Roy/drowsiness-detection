import cv2
import mediapipe as mp

# Initialize MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

# Open the webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Convert the image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image and get the landmarks
    result = face_mesh.process(image_rgb)

    # Draw face landmarks on the image
    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(image, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION)

    # Display the image
    cv2.imshow('Face Mesh', image)

    if cv2.waitKey(5) & 0xFF == 27:  # Press ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
