import config
import streamlit as st
import time
import numpy as np
import cv2
import mediapipe as mp
import tensorflow as tf
import joblib
import os
import json
import threading
from queue import Queue
import pyttsx3
import pandas as pd
import math

# Page configuration
st.set_page_config(
    page_title="Yoga Pose Classification",
    page_icon="🧘‍♀️",
    layout="wide"
)

class RealtimePoseClassifier:
    def __init__(self, model_dir='Streamlit-application/model'):
        try:
            # Load model components
            self.model = tf.keras.models.load_model(os.path.join(model_dir, 'upward_salute_pose_classifier_v4.0.h5'))
            self.preprocessors = joblib.load(os.path.join(model_dir, 'preprocessors.joblib'))
            self.standing = False
            self.color = (255, 255, 255)
            self.current_pose = None  # Track current pose
            self.current_confidence = 0.0 
            
            # MediaPipe setup
            self.mp_pose = mp.solutions.pose
            self.mp_drawing = mp.solutions.drawing_utils
        except Exception as e:
            raise Exception(f"Failed to initialize classifier: {str(e)}")
        # Load metadata
        with open(os.path.join(model_dir, 'model_metadata.json'), 'r') as f:
            self.metadata = json.load(f)

        self.visibility_threshold = 0.5
        self.total_landmarks = 33
        self.essential_landmarks = [
            self.mp_pose.PoseLandmark.LEFT_SHOULDER.value,
            self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
            self.mp_pose.PoseLandmark.LEFT_ELBOW.value,
            self.mp_pose.PoseLandmark.RIGHT_ELBOW.value,
            self.mp_pose.PoseLandmark.LEFT_WRIST.value,
            self.mp_pose.PoseLandmark.RIGHT_WRIST.value,
            self.mp_pose.PoseLandmark.LEFT_HIP.value,
            self.mp_pose.PoseLandmark.RIGHT_HIP.value,
            self.mp_pose.PoseLandmark.LEFT_KNEE.value,
            self.mp_pose.PoseLandmark.RIGHT_KNEE.value,
            self.mp_pose.PoseLandmark.LEFT_ANKLE.value,
            self.mp_pose.PoseLandmark.RIGHT_ANKLE.value
        ]

        # Speech queue setup
        self.speech_queue = Queue()
        self.speech_thread = threading.Thread(target=self._speech_worker, daemon=True)
        self.speech_thread.start()
        self.last_message = ""
        self.message_cooldown = 7.0
        self.last_message_time = 0

        # Reference video setup
        video_path = 'C:/Users/DELL/YOJE/Services/data/upstand1.mp4'
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Reference video not found at {video_path}")
            
        self.reference_video = cv2.VideoCapture(video_path)
        if not self.reference_video.isOpened():
            raise ValueError("Could not open reference video")
            
        self.reference_frame = None
        self.ref_width = int(self.reference_video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.ref_height = int(self.reference_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.ref_display_width = 200
        self.ref_display_height = 150

    def update_reference_frame(self):
        ret, frame = self.reference_video.read()
        if not ret:
            self.reference_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.reference_video.read()
        
        if ret:
            self.reference_frame = cv2.resize(frame, (self.ref_display_width, self.ref_display_height))
            return True
        return False

    def overlay_reference_video(self, image):
        if self.reference_frame is None:
            return image

        y_offset = image.shape[0] - self.ref_display_height - 10
        x_offset = image.shape[1] - self.ref_display_width - 10

        try:
            cv2.rectangle(image, 
                        (x_offset-5, y_offset-5),
                        (x_offset + self.ref_display_width+5, y_offset + self.ref_display_height+5),
                        (245, 117, 16),
                        -1)
            
            image[y_offset:y_offset+self.ref_display_height, 
                 x_offset:x_offset+self.ref_display_width] = self.reference_frame
                
        except Exception as e:
            st.error(f"Error overlaying reference video: {e}")
        
        return image

    def _speech_worker(self):
        while True:
            message = self.speech_queue.get()
            pyttsx3.speak(message)
            self.speech_queue.task_done()

    def queue_speech(self, message):
        current_time = time.time()
        if current_time - self.last_message_time >= self.message_cooldown:
            self.speech_queue.put(message)
            self.last_message = message
            self.last_message_time = current_time

    def check_visibility(self, landmarks):
        missing = []
        for idx in self.essential_landmarks:
            if landmarks[idx].visibility < self.visibility_threshold:
                missing.append(self.mp_pose.PoseLandmark(idx).name)
        return len(missing) == 0, missing

    def calculate_angle(self, landmark1, landmark2, landmark3):
        x1, y1, _ = landmark1
        x2, y2, _ = landmark2
        x3, y3, _ = landmark3

        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
        return angle + 360 if angle < 0 else angle

    def check_angles(self, results, pred_class):
        try:
            # Calculate angles
            left_elbow_angle = self.calculate_angle(
                [results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                 results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y,
                 results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].z],
                [results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                 results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].y,
                 results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].z],
                [results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                 results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST.value].y,
                 results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST.value].z]
            )
            
            right_elbow_angle = self.calculate_angle(
                [results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                 results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y,
                 results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].z],
                [results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                 results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].y,
                 results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].z],
                [results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                 results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].y,
                 results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].z]
            )

            left_knee_angle = self.calculate_angle(
                [results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP.value].x,
                 results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP.value].y,
                 results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP.value].z],
                [results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                 results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_KNEE.value].y,
                 results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_KNEE.value].z],
                [results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                 results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].y,
                 results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].z]
            )

            right_knee_angle = self.calculate_angle(
                [results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                 results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP.value].y,
                 results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP.value].z],
                [results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                 results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].y,
                 results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].z],
                [results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                 results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value].y,
                 results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value].z]
            )


            if not (165 < left_knee_angle < 195 and 165 < right_knee_angle < 195):
                self.queue_speech("Please stand straight")
                self.standing = False
                return False, "Please stand straight"

            if pred_class in ["front", "up"]:
                if not (165 < left_elbow_angle < 195 and 165 < right_elbow_angle < 195):
                    self.queue_speech("Please Keep your hands straight")
                    return False, "Keep your hands straight"

            self.standing = True
            return True, "Correct pose"

        except Exception as e:
            return False, f"Error calculating angles: {e}"

    def run_detection(self):
        st.title("Yoga Pose Classification")
    
        # Subtitle and instructions using markdown hierarchy
        st.markdown("""
            * Upward Salute (Urdhva Hastasana)
            * Follow the reference video to perform the pose correctly.
            * Don't forget to give Feedback!!
        """)
        
        # Create layout
        col1, col2 = st.columns([3, 1])
        
        # Main video feed column
        with col1:
            frame_window = st.image([])
            status_text = st.empty()
        
        # Control buttons
        start_button = st.button("Start Camera")
        stop_button = st.button("Stop")

        # Initialize and start camera feed
        if start_button and not stop_button:
            cap = cv2.VideoCapture(0)
            pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

            # Reference pose column
            with col2:
                reference_window = st.image([])
                pose_container = st.empty()

                # Show initial reference frame
                if self.update_reference_frame():
                    reference_window.image(self.reference_frame, channels="RGB")

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Process frame
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = pose.process(image)
                image.flags.writeable = True

                try:
                    if not results.pose_landmarks:
                        status_text.error('No body detected')
                        self.queue_speech("No body detected")
                    else:
                        # Draw landmarks
                        self.mp_drawing.draw_landmarks(
                            image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS
                        )

                        # Check visibility
                        all_visible, missing = self.check_visibility(results.pose_landmarks.landmark)
                        
                        if not all_visible:
                            status_text.warning('Please bring your entire body in frame')
                            self.queue_speech("Please bring your entire body in frame")
                        else:
                            # Extract landmarks
                            landmarks = []
                            for landmark in results.pose_landmarks.landmark:
                                landmarks.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])

                            # Make prediction
                            X = pd.DataFrame([landmarks])
                            X_scaled = self.preprocessors['scaler'].transform(X)
                            pred = self.model.predict(X_scaled, verbose=0)
                            pred_class = self.preprocessors['label_encoder'].inverse_transform([np.argmax(pred)])[0]
                            pred_prob = float(np.max(pred))

                            # Update pose only if it changed significantly
                            if pred_class != self.current_pose or abs(pred_prob - self.current_confidence) > 0.1:
                                self.current_pose = pred_class
                                self.current_confidence = pred_prob
                                pose_container.success(f"""
                                    ### Current Pose
                                    **Class:** {self.current_pose}  
                                    **Confidence:** {self.current_confidence:.2f}
                                """)

                            # Check angles and update reference
                            angles_ok, message = self.check_angles(results, pred_class)
                            if self.update_reference_frame():
                                reference_window.image(
                                    self.reference_frame, 
                                    channels="RGB",
                                    caption="Reference: Upward Salute Pose"
                                )

                            # Update status
                            if angles_ok:
                                status_text.success("✅ Correct pose!")
                            else:
                                status_text.warning(f"⚠️ {message}")

                except Exception as e:
                    status_text.error(f"Error: {str(e)}")
                    print(f"Error in pose detection: {str(e)}")

                # Display the processed frame
                frame_window.image(image, channels="RGB")

                # Check for stop button
                if stop_button:
                    break

            # Cleanup
            cap.release()
            status_text.empty()
            frame_window.empty()
            reference_window.empty()
            pose_container.empty()

def show_user_page():
    # Initialize session state for feedback if not exists
    if 'feedback' not in st.session_state:
        st.session_state.feedback = ""

    try:
        classifier = RealtimePoseClassifier()
        classifier.run_detection()
        
        # Move feedback collection to a function that returns the feedback
        feedback = collect_feedback()
        return feedback
        
    except Exception as e:
        st.error(f"Error initializing classifier: {e}")
        return None

def collect_feedback():
    """Collect and return user feedback"""
    feedback = ""
    with st.sidebar:
        st.title("Settings")
        st.markdown("## Feedback")
        feedback = st.text_area("Share your feedback:", key="user_feedback")
        if st.button("Submit Feedback"):
            st.success("Thank you for your feedback!")
            st.session_state.feedback = feedback
    return st.session_state.feedback

def main():
    try:
        classifier = RealtimePoseClassifier()
        classifier.run_detection()
        with st.sidebar:
            st.title("Settings")
            st.markdown("## Feedback")
            feedback = st.text_area("Share your feedback:")
            if st.button("Submit Feedback"):
                st.success("Thank you for your feedback!")
            return feedback
    except Exception as e:
        st.error(f"Error initializing classifier: {e}")

if __name__ == "__main__":
    show_user_page()
