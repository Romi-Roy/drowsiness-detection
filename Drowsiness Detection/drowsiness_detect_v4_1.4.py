import cv2
import time
import numpy as np
import mediapipe as mp
import pygame
import firebase_admin
from firebase_admin import credentials, db
from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates as denormalize_coordinates
from datetime import datetime
import pytz 
from mediapipe.python.solutions.face_mesh_connections import FACEMESH_TESSELATION

# Firebase setup
cred = credentials.Certificate('safe-drive-f82b1-firebase-adminsdk-np4sd-e425a9e585.json')  # Path to your Firebase SDK JSON
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://safe-drive-f82b1-default-rtdb.asia-southeast1.firebasedatabase.app/'  # Your Firebase database URL
})

# Initialize pygame for sound
pygame.mixer.init()
pygame.mixer.music.load('alarm.wav')

# Mediapipe and drawing utilities
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def get_mediapipe_app(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5):
    """Initialize Mediapipe FaceMesh."""
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=max_num_faces,
        refine_landmarks=refine_landmarks,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )
    return face_mesh

def get_ist_timestamp():
    """Return current timestamp in Indian Standard Time."""
    ist = pytz.timezone('Asia/Kolkata')
    return datetime.now(ist).strftime('%Y-%m-%dT%H:%M:%S')

def distance(point_1, point_2):
    """Calculate l2-norm between two points."""
    return sum([(i - j) ** 2 for i, j in zip(point_1, point_2)]) ** 0.5

def get_ear(landmarks, refer_idxs, frame_width, frame_height):
    """Calculate Eye Aspect Ratio for one eye."""
    try:
        coords_points = []
        for i in refer_idxs:
            lm = landmarks[i]
            coord = denormalize_coordinates(lm.x, lm.y, frame_width, frame_height)
            coords_points.append(coord)

        P2_P6 = distance(coords_points[1], coords_points[5])
        P3_P5 = distance(coords_points[2], coords_points[4])
        P1_P4 = distance(coords_points[0], coords_points[3])

        ear = (P2_P6 + P3_P5) / (2.0 * P1_P4)
    except:
        ear = 0.0
        coords_points = None

    return ear, coords_points

def calculate_avg_ear(landmarks, left_eye_idxs, right_eye_idxs, image_w, image_h):
    left_ear, left_lm_coordinates = get_ear(landmarks, left_eye_idxs, image_w, image_h)
    right_ear, right_lm_coordinates = get_ear(landmarks, right_eye_idxs, image_w, image_h)
    avg_ear = (left_ear + right_ear) / 2.0
    return avg_ear, (left_lm_coordinates, right_lm_coordinates)

def detect_head_nodding(landmarks, threshold):
    """Detect head nodding using face landmarks (e.g., points on the nose and chin)."""
    nose_tip = landmarks[1]  # Example landmark for nose tip
    chin_tip = landmarks[152]  # Example landmark for chin tip
    
    vertical_distance = abs(nose_tip.y - chin_tip.y)
    if vertical_distance > threshold:
        return True  # Head nod detected
    return False

class VideoFrameHandler:
    def __init__(self):
        self.eye_idxs = {
            "left": [362, 385, 387, 263, 373, 380],
            "right": [33, 160, 158, 133, 153, 144],
        }
        self.RED = (0, 0, 255)  # BGR
        self.GREEN = (0, 255, 0)  # BGR
        self.facemesh_model = get_mediapipe_app()
        self.state_tracker = {
            "start_time": time.perf_counter(),
            "DROWSY_TIME": 0.0,
            "COLOR": self.GREEN,
            "play_alarm": False,
            "state": "AWAKE"
        }

    def update_firebase(self, state, ear, user_id="user_123"):
        """Update Firebase with the current drowsiness state and details."""
        ref = db.reference('drowsiness_data')
        ref.set({
            "state": state,
            "timestamp": get_ist_timestamp(),
            "user_id": user_id,
            "details": {
                "eye_aspect_ratio": ear,
                "alertness_score": 80  # Example score
            }
        })

    def play_alarm(self):
        """Play the alarm sound."""
        if not pygame.mixer.music.get_busy():
            pygame.mixer.music.play(-1)

    def stop_alarm(self):
        """Stop the alarm sound."""
        pygame.mixer.music.stop()

    def process(self, frame, thresholds):
        frame.flags.writeable = False
        frame_h, frame_w, _ = frame.shape

        DROWSY_TIME_txt_pos = (10, int(frame_h // 2 * 1.7))
        ALM_txt_pos = (10, int(frame_h // 2 * 1.85))
        EAR_txt_pos = (10, 30)

        results = self.facemesh_model.process(frame)
        frame.flags.writeable = True

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            ear, coordinates = calculate_avg_ear(landmarks, self.eye_idxs["left"], self.eye_idxs["right"], frame_w, frame_h)
            head_nod = detect_head_nodding(landmarks, thresholds["HEAD_NOD_THRESH"])

            # Draw face mesh with red color
            mp_drawing.draw_landmarks(
                frame, 
                results.multi_face_landmarks[0],
                FACEMESH_TESSELATION,  # Replace mp_face_mesh.FACE_CONNECTIONS with FACEMESH_TESSELATION
                mp_drawing.DrawingSpec(color=self.RED, thickness=1, circle_radius=1),
                mp_drawing.DrawingSpec(color=self.RED, thickness=1)
            )

            # Draw green eye points
            if coordinates:
                for eye_coordinates in coordinates:
                    if eye_coordinates:
                        for point in eye_coordinates:
                            if point:
                                cv2.circle(frame, point, 2, self.GREEN, -1)

            if ear < thresholds["EAR_THRESH"] or head_nod:
                end_time = time.perf_counter()
                self.state_tracker["DROWSY_TIME"] += end_time - self.state_tracker["start_time"]
                self.state_tracker["start_time"] = end_time
                self.state_tracker["COLOR"] = self.RED

                if self.state_tracker["DROWSY_TIME"] >= thresholds["WAIT_TIME"]:
                    self.state_tracker["play_alarm"] = True
                    self.state_tracker["state"] = "EXTREMELY_SLEEPY" if self.state_tracker["DROWSY_TIME"] >= 3 else "SLEEPY"
                    self.play_alarm()
                    plot_text(frame, "WAKE UP! WAKE UP", ALM_txt_pos, self.state_tracker["COLOR"])
                    self.update_firebase(self.state_tracker["state"], ear)

            else:
                self.state_tracker["start_time"] = time.perf_counter()
                self.state_tracker["DROWSY_TIME"] = 0.0
                self.state_tracker["COLOR"] = self.GREEN
                self.state_tracker["play_alarm"] = False
                self.state_tracker["state"] = "AWAKE"
                self.stop_alarm()
                self.update_firebase("AWAKE", ear)

            # Display EAR and DROWSY time
            EAR_txt = f"EAR: {round(ear, 2)}"
            DROWSY_TIME_txt = f"DROWSY: {round(self.state_tracker['DROWSY_TIME'], 3)} Secs"
            plot_text(frame, EAR_txt, EAR_txt_pos, self.state_tracker["COLOR"])
            plot_text(frame, DROWSY_TIME_txt, DROWSY_TIME_txt_pos, self.state_tracker["COLOR"])

        else:
            self.state_tracker["start_time"] = time.perf_counter()
            self.state_tracker["DROWSY_TIME"] = 0.0
            self.state_tracker["COLOR"] = self.GREEN
            self.state_tracker["play_alarm"] = False
            self.stop_alarm()

        return frame

def plot_text(frame, text, pos, color):
    cv2.putText(
        frame, text, pos, cv2.FONT_HERSHEY_DUPLEX, 1, color, 2, cv2.LINE_AA
    )

def main():
    """Main driver code."""
    # Replace the `0` with the path to the video file.
    video_path = 'video_inputs/video_002.mp4' # Replace this with your video file path
    cap = cv2.VideoCapture(video_path)
    handler = VideoFrameHandler()

    thresholds = {
        "EAR_THRESH": 0.21,  # Lower EAR threshold for drowsiness detection
        "WAIT_TIME": 1.5,  # Time (in seconds) after which alarm is triggered
        "HEAD_NOD_THRESH": 0.3  # Threshold for detecting head nodding
    }

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error reading frame or end of video reached.")
            break

        frame = cv2.flip(frame, 1)  # Flip frame horizontally (optional, based on video orientation)
        frame = handler.process(frame, thresholds)

        cv2.imshow("Drowsiness Detection", frame)
        if cv2.waitKey(5) & 0xFF == 27:  # Press 'Esc' to exit the loop
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
