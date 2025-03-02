import config
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
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

class PoseProcessor(VideoProcessorBase):
    def __init__(self, model_dir='model'):
        self.classifier = RealtimePoseClassifier(model_dir)
        self.current_pose = None
        self.current_confidence = 0.0
        self.status_message = "Initializing..."
        self.status_color = "white"

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Process frame using classifier
        img, pose_class, confidence, message, color = self.classifier.process_frame(img)
        
        # Update status
        self.current_pose = pose_class
        self.current_confidence = confidence
        self.status_message = message
        self.status_color = color
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

class RealtimePoseClassifier:
    def __init__(self, model_dir='model'):
        try:
            # Load model components
            self.model = tf.keras.models.load_model(os.path.join(model_dir, 'upward_salute_pose_classifier_v4.0.h5'))
            self.preprocessors = joblib.load(os.path.join(model_dir, 'preprocessors.joblib'))
            self.standing = False
            self.color = (255, 255, 255)
            self.current_pose = None  # Track current pose
            self.current_confidence = 0.0 
            
            # MediaPipe setup - Fixed the initialization
            self.mp_pose = mp.solutions.pose
            self.mp_drawing = mp.solutions.drawing_utils
            self.pose = self.mp_pose.Pose(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        except Exception as e:
            st.error(f"Failed to initialize classifier: {str(e)}")
            raise Exception(f"Failed to initialize classifier: {str(e)}")
            
        try:
            # Load metadata
            with open(os.path.join(model_dir, 'model_metadata.json'), 'r') as f:
                self.metadata = json.load(f)
        except Exception as e:
            st.warning(f"Could not load model metadata: {str(e)}")
            self.metadata = {"classes": ["front", "up"]}

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
        try:
            video_path = os.path.join(os.path.dirname(model_dir), 'data', 'upstand1.mp4')
            if not os.path.exists(video_path):
                st.warning(f"Reference video not found at {video_path}. Using static image instead.")
                self.reference_video = None
                self.reference_frame = np.zeros((150, 200, 3), dtype=np.uint8)
                cv2.putText(self.reference_frame, "Reference", (50, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            else:
                self.reference_video = cv2.VideoCapture(video_path)
                if not self.reference_video.isOpened():
                    st.warning("Could not open reference video. Using static image instead.")
                    self.reference_video = None
                    self.reference_frame = np.zeros((150, 200, 3), dtype=np.uint8)
                    cv2.putText(self.reference_frame, "Reference", (50, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                else:
                    self.reference_frame = None
                    self.ref_width = int(self.reference_video.get(cv2.CAP_PROP_FRAME_WIDTH))
                    self.ref_height = int(self.reference_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        except Exception as e:
            st.warning(f"Error setting up reference video: {str(e)}")
            self.reference_video = None
            self.reference_frame = np.zeros((150, 200, 3), dtype=np.uint8)
            cv2.putText(self.reference_frame, "Reference", (50, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
        self.ref_display_width = 200
        self.ref_display_height = 150

    def update_reference_frame(self):
        if self.reference_video is None:
            return True
            
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
            # Create a copy of the image to avoid modification errors
            result = image.copy()
            
            # Add background rectangle
            cv2.rectangle(result, 
                        (x_offset-5, y_offset-5),
                        (x_offset + self.ref_display_width+5, y_offset + self.ref_display_height+5),
                        (245, 117, 16),
                        -1)
            
            # Create ROI and overlay reference frame
            roi = result[y_offset:y_offset+self.ref_display_height, 
                        x_offset:x_offset+self.ref_display_width]
            
            # Ensure ROI and reference frame have the same shape
            if roi.shape == self.reference_frame.shape:
                result[y_offset:y_offset+self.ref_display_height, 
                      x_offset:x_offset+self.ref_display_width] = self.reference_frame
            else:
                # Resize reference frame if shapes don't match
                resized_ref = cv2.resize(self.reference_frame, (roi.shape[1], roi.shape[0]))
                result[y_offset:y_offset+roi.shape[0], 
                      x_offset:x_offset+roi.shape[1]] = resized_ref
                
            return result
                
        except Exception as e:
            st.warning(f"Error overlaying reference video: {e}")
            return image

    def _speech_worker(self):
        while True:
            message = self.speech_queue.get()
            try:
                engine = pyttsx3.init()
                engine.say(message)
                engine.runAndWait()
            except Exception as e:
                print(f"Speech error: {e}")
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

    def process_frame(self, frame):
        """Process a single frame and return the result"""
        # Convert to RGB for processing
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        # Process with MediaPipe
        results = self.pose.process(image)
        
        # Convert back for display
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        pose_class = None
        confidence = 0.0
        status_message = "No body detected"
        status_color = (0, 0, 255)  # Red
        
        # Update reference frame
        self.update_reference_frame()
        
        try:
            if not results.pose_landmarks:
                self.queue_speech("No body detected")
            else:
                # Draw landmarks
                self.mp_drawing.draw_landmarks(
                    image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS
                )

                # Check visibility
                all_visible, missing = self.check_visibility(results.pose_landmarks.landmark)
                
                if not all_visible:
                    status_message = "Please bring your entire body in frame"
                    status_color = (0, 165, 255)  # Orange
                    self.queue_speech(status_message)
                else:
                    # Extract landmarks
                    landmarks = []
                    for landmark in results.pose_landmarks.landmark:
                        landmarks.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])

                    # Make prediction
                    X = pd.DataFrame([landmarks])
                    X_scaled = self.preprocessors['scaler'].transform(X)
                    pred = self.model.predict(X_scaled, verbose=0)
                    pose_class = self.preprocessors['label_encoder'].inverse_transform([np.argmax(pred)])[0]
                    confidence = float(np.max(pred))

                    # Check angles
                    angles_ok, message = self.check_angles(results, pose_class)
                    
                    if angles_ok:
                        status_message = f"Correct pose: {pose_class} ({confidence:.2f})"
                        status_color = (0, 255, 0)  # Green
                    else:
                        status_message = message
                        status_color = (0, 165, 255)  # Orange
                        
        except Exception as e:
            status_message = f"Error: {str(e)}"
            status_color = (0, 0, 255)  # Red
        
        # Draw status on image
        cv2.rectangle(image, (0, 0), (image.shape[1], 40), (0, 0, 0), -1)
        cv2.putText(image, status_message, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        # Overlay reference video
        image = self.overlay_reference_video(image)
            
        return image, pose_class, confidence, status_message, status_color


def main():
    st.title("Yoga Pose Classification")

    # Subtitle and instructions using markdown
    st.markdown("""
        * Upward Salute (Urdhva Hastasana)
        * Follow the reference video to perform the pose correctly.
        * Don't forget to give Feedback!!
    """)
    
    # Create layout
    col1, col2 = st.columns([3, 1])
    
    # Create a placeholder for status
    status_placeholder = st.empty()
    
    # RTC configuration
    rtc_config = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
    
    # WebRTC streamer component
    webrtc_ctx = webrtc_streamer(
        key="yoga-pose-classification",
        mode=webrtc_streamer.RENDER_MODE,
        video_processor_factory=PoseProcessor,
        rtc_configuration=rtc_config,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
    
    # Display current pose information
    if webrtc_ctx.video_processor:
        with col2:
            st.subheader("Current Pose")
            pose_info = st.empty()
            
            # Update pose information periodically
            if webrtc_ctx.state.playing:
                while True:
                    processor = webrtc_ctx.video_processor
                    
                    # Update info if available
                    if processor.current_pose:
                        pose_info.success(f"""
                            ### Classification
                            **Pose:** {processor.current_pose}  
                            **Confidence:** {processor.current_confidence:.2f}
                        """)
                    else:
                        pose_info.warning("Waiting for detection...")
                    
                    # Update status message
                    if processor.status_color == (0, 255, 0):  # Green
                        status_placeholder.success(f"✅ {processor.status_message}")
                    elif processor.status_color == (0, 165, 255):  # Orange
                        status_placeholder.warning(f"⚠️ {processor.status_message}")
                    else:
                        status_placeholder.error(f"❌ {processor.status_message}")
                    
                    # Sleep to reduce CPU usage
                    time.sleep(0.1)
    
    # Feedback section
    with st.sidebar:
        st.title("Settings")
        st.markdown("## Feedback")
        feedback = st.text_area("Share your feedback:")
        if st.button("Submit Feedback"):
            st.success("Thank you for your feedback!")

if __name__ == "__main__":
    main()
