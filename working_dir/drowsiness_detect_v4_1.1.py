import cv2
import time
import numpy as np
import mediapipe as mp
import pygame
from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates as denormalize_coordinates

def get_mediapipe_app(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5):
    """Initialize and return Mediapipe FaceMesh Solution Graph object"""
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=max_num_faces,
        refine_landmarks=refine_landmarks,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )
    return face_mesh

def distance(point_1, point_2):
    """Calculate l2-norm between two points"""
    dist = sum([(i - j) ** 2 for i, j in zip(point_1, point_2)]) ** 0.5
    return dist

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
        
        # Initialize pygame for sound
        pygame.mixer.init()
        pygame.mixer.music.load('alarm.wav')

    def play_alarm(self):
        """Play the alarm sound, restarting if already playing."""
        if not pygame.mixer.music.get_busy():  # Check if music is currently playing
            pygame.mixer.music.play(-1)  # Play in loop

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

            # Plot eye landmarks
            for lm_coordinates in coordinates:
                if lm_coordinates:
                    for coord in lm_coordinates:
                        cv2.circle(frame, coord, 2, self.state_tracker["COLOR"], -1)

            if ear < thresholds["EAR_THRESH"]:
                end_time = time.perf_counter()
                self.state_tracker["DROWSY_TIME"] += end_time - self.state_tracker["start_time"]
                self.state_tracker["start_time"] = end_time
                self.state_tracker["COLOR"] = self.RED

                if self.state_tracker["DROWSY_TIME"] >= thresholds["WAIT_TIME"]:
                    self.state_tracker["play_alarm"] = True
                    self.play_alarm()  # Play alarm sound
                    plot_text(frame, "WAKE UP! WAKE UP", ALM_txt_pos, self.state_tracker["COLOR"])

            else:
                self.state_tracker["start_time"] = time.perf_counter()
                self.state_tracker["DROWSY_TIME"] = 0.0
                self.state_tracker["COLOR"] = self.GREEN
                self.state_tracker["play_alarm"] = False
                self.stop_alarm()  # Stop alarm sound

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
            self.stop_alarm()  # Stop alarm sound

            # Flip the frame horizontally for a selfie-view display.
            frame = cv2.flip(frame, 1)

        return frame, self.state_tracker["play_alarm"]

def plot_text(image, text, origin, color, font=cv2.FONT_HERSHEY_SIMPLEX, fntScale=0.8, thickness=2):
    """Plot text on the image."""
    image = cv2.putText(image, text, origin, font, fntScale, color, thickness)
    return image

if __name__ == "__main__":
    try:
        cap = cv2.VideoCapture(0)  # Attempt to access the default webcam
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            exit()

        video_handler = VideoFrameHandler()
        thresholds = {
            "EAR_THRESH": 0.18,  # Example threshold for Eye Aspect Ratio
            "WAIT_TIME": 1.0     # Example wait time before sounding the alarm
        }

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame from webcam.")
                break

            processed_frame, play_alarm = video_handler.process(frame, thresholds)

            cv2.imshow('Drowsiness Detection', processed_frame)

            # Press 'q' to exit the loop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"An error occurred: {e}")
