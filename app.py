import streamlit as st
import pandas as pd
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
from PIL import Image
from fer import FER
import cv2
import mediapipe as mp
import tempfile
from collections import Counter

# Initialize the FER detector
detector = FER(mtcnn=True)
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Initialize session state
if 'current_selection' not in st.session_state:
    st.session_state.current_selection = None
if 'webcam_active' not in st.session_state:
    st.session_state.webcam_active = False
if 'webcam_results' not in st.session_state:
    st.session_state.webcam_results = []

# Set the title of the dashboard
st.title("Student Engagement Dashboard")

# Sidebar options for real-time webcam stream, photo, and video uploads
st.sidebar.header("Options")

if st.sidebar.button("Start Webcam"):
    st.session_state.current_selection = "webcam"
    st.session_state.webcam_active = True

st.sidebar.subheader("Upload Student Photos")
uploaded_photo = st.sidebar.file_uploader("Choose a photo...", type=["jpg", "jpeg", "png"], key="photo_uploader")

st.sidebar.subheader("Upload Student Videos")
uploaded_video = st.sidebar.file_uploader("Choose a video...", type=["mp4", "mov", "avi"], key="video_uploader")

if st.sidebar.button("Back to Dashboard"):
    st.session_state.current_selection = None
    st.session_state.webcam_active = False
    st.session_state.webcam_results = []

# Function to determine the engagement state based on the uploaded photo
def detect_engagement_state(photo):
    image = Image.open(photo).convert("RGB")  # Convert image to RGB
    image = np.array(image)

    # Detect emotions
    emotions = detector.detect_emotions(image)
    engagement_state = "not focused"  # Default state

    if emotions:
        top_emotion = max(emotions[0]["emotions"], key=emotions[0]["emotions"].get)
        emotion_confidence = emotions[0]["emotions"][top_emotion]

        # Placeholder logic for engagement state
        if top_emotion == "happy" and emotion_confidence > 0.7:
            engagement_state = "focused"
        elif top_emotion == "neutral" and emotion_confidence > 0.7:
            engagement_state = "focused"
        elif top_emotion == "sad" and emotion_confidence > 0.7:
            engagement_state = "not focused"
        elif top_emotion == "angry" and emotion_confidence > 0.7:
            engagement_state = "not focused"

    # Detect poses using MediaPipe
    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        if results.pose_landmarks:
            # Check for raised hand
            left_hand_raised = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y < results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y
            right_hand_raised = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y < results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y
            
            if left_hand_raised or right_hand_raised:
                engagement_state = "raise_hand"

    return engagement_state

# Function to analyze video frames
def analyze_video_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    engagement_results = []

    with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5) as pose:
        for i in range(frame_count):
            ret, frame = cap.read()
            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            emotions = detector.detect_emotions(image)
            engagement_state = "not focused"  # Default state

            if emotions:
                top_emotion = max(emotions[0]["emotions"], key=emotions[0]["emotions"].get)
                emotion_confidence = emotions[0]["emotions"][top_emotion]

                # Placeholder logic for engagement state
                if top_emotion == "happy" and emotion_confidence > 0.7:
                    engagement_state = "focused"
                elif top_emotion == "neutral" and emotion_confidence > 0.7:
                    engagement_state = "focused"
                elif top_emotion == "sad" and emotion_confidence > 0.7:
                    engagement_state = "not focused"
                elif top_emotion == "angry" and emotion_confidence > 0.7:
                    engagement_state = "not focused"

            # Detect poses using MediaPipe
            results = pose.process(image)
            if results.pose_landmarks:
                # Check for raised hand
                left_hand_raised = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y < results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y
                right_hand_raised = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y < results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y

                if left_hand_raised or right_hand_raised:
                    engagement_state = "raise_hand"

            engagement_results.append((i / frame_rate, engagement_state))
    
    cap.release()
    return engagement_results

# Function to analyze webcam frames
def analyze_webcam_frame(image):
    emotions = detector.detect_emotions(image)
    engagement_state = "not focused"  # Default state

    if emotions:
        top_emotion = max(emotions[0]["emotions"], key=emotions[0]["emotions"].get)
        emotion_confidence = emotions[0]["emotions"][top_emotion]

        # Placeholder logic for engagement state
        if top_emotion == "happy" and emotion_confidence > 0.7:
            engagement_state = "focused"
        elif top_emotion == "neutral" and emotion_confidence > 0.7:
            engagement_state = "focused"
        elif top_emotion == "sad" and emotion_confidence > 0.7:
            engagement_state = "not focused"
        elif top_emotion == "angry" and emotion_confidence > 0.7:
            engagement_state = "not focused"

    # Detect poses using MediaPipe
    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        if results.pose_landmarks:
            # Check for raised hand
            left_hand_raised = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y < results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y
            right_hand_raised = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y < results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y
            
            if left_hand_raised or right_hand_raised:
                engagement_state = "raise_hand"

    return engagement_state

# Function to summarize engagement results
def summarize_engagement_results(engagement_results, min_consecutive_frames=5):
    summary = []
    current_state = engagement_results[0][1]
    current_start = engagement_results[0][0]
    current_count = 1

    for i in range(1, len(engagement_results)):
        time, state = engagement_results[i]
        if state == current_state:
            current_count += 1
        else:
            if current_count >= min_consecutive_frames:
                summary.append((current_start, engagement_results[i-1][0], current_state))
            current_state = state
            current_start = time
            current_count = 1

    if current_count >= min_consecutive_frames:
        summary.append((current_start, engagement_results[-1][0], current_state))

    return summary

# Function to convert engagement results to DataFrame
def engagement_results_to_df(engagement_results):
    df = pd.DataFrame(engagement_results, columns=["Time", "Engagement State"])
    return df

# RTC Configuration for better webcam handling
RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

if st.session_state.current_selection == "photo":
    st.header("Uploaded Photo")
    st.image(uploaded_photo, caption='Uploaded Photo', use_column_width=True)
    engagement_state = detect_engagement_state(uploaded_photo)
    st.success(f"The student appears to be: {engagement_state}")
    st.success("Photo uploaded successfully!")

if st.session_state.current_selection == "video":
    st.header("Uploaded Video")
    st.video(uploaded_video)

    # Analyze video frames
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_video.read())
        video_path = tmp_file.name

    engagement_results = []

    # Progress bar
    cap = cv2.VideoCapture(video_path)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_bar = st.progress(0)

    with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5) as pose:
        for i in range(frame_count):
            ret, frame = cap.read()
            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            emotions = detector.detect_emotions(image)
            engagement_state = "not focused"  # Default state

            if emotions:
                top_emotion = max(emotions[0]["emotions"], key=emotions[0]["emotions"].get)
                emotion_confidence = emotions[0]["emotions"][top_emotion]

                # Placeholder logic for engagement state
                if top_emotion == "happy" and emotion_confidence > 0.7:
                    engagement_state = "focused"
                elif top_emotion == "neutral" and emotion_confidence > 0.7:
                    engagement_state = "focused"
                elif top_emotion == "sad" and emotion_confidence > 0.7:
                    engagement_state = "not focused"
                elif top_emotion == "angry" and emotion_confidence > 0.7:
                    engagement_state = "not focused"

            # Detect poses using MediaPipe
            results = pose.process(image)
            if results.pose_landmarks:
                # Check for raised hand
                left_hand_raised = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y < results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y
                right_hand_raised = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y < results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y

                if left_hand_raised or right_hand_raised:
                    engagement_state = "raise_hand"

            engagement_results.append((i / frame_rate, engagement_state))
            progress_bar.progress(int((i / frame_count) * 100))
    
    cap.release()

    summary = summarize_engagement_results(engagement_results)

    # Convert engagement results to DataFrame
    engagement_df = engagement_results_to_df(engagement_results)

    # Display analysis results
    st.header("Engagement Summary")
    for start, end, state in summary:
        st.write(f"From {start:.1f}s to {end:.1f}s: {state}")
    st.success("Video uploaded and analyzed successfully!")

    # Display engagement data as a line chart
    st.header("Real-Time Engagement Data")
    engagement_counts = engagement_df.groupby(["Time", "Engagement State"]).size().unstack(fill_value=0)
    st.line_chart(engagement_counts)

if st.session_state.current_selection == "webcam" and st.session_state.webcam_active:
    st.header("Webcam Stream")
    
    class VideoTransformer(VideoTransformerBase):
        def __init__(self):
            self.pose = mp.solutions.pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
        
        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")
            engagement_state = analyze_webcam_frame(img)
            st.session_state.webcam_results.append((st.session_state.webcam_results.__len__() / 30, engagement_state))
            
            # Annotate the frame
            results = self.pose.process(img)
            mp_drawing.draw_landmarks(img, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)

            return img

    webrtc_streamer(key="webcam_stream", video_transformer_factory=VideoTransformer, rtc_configuration=RTC_CONFIGURATION, media_stream_constraints={"video": True, "audio": False})

if st.session_state.current_selection == "webcam":
    if st.session_state.webcam_results:
        # Summarize webcam results
        webcam_summary = summarize_engagement_results(st.session_state.webcam_results)
        
        # Display webcam engagement summary
        st.header("Webcam Engagement Summary")
        for start, end, state in webcam_summary:
            st.write(f"From {start:.1f}s to {end:.1f}s: {state}")
        
        # Convert webcam results to DataFrame and display as a line chart
        webcam_df = engagement_results_to_df(st.session_state.webcam_results)
        st.header("Real-Time Webcam Engagement Data")
        webcam_counts = webcam_df.groupby(["Time", "Engagement State"]).size().unstack(fill_value=0)
        st.line_chart(webcam_counts)

# Display the Real-Time Engagement Data and Engagement Metrics only when a video or photo is uploaded
if st.session_state.current_selection in ["video", "photo"]:

    # Additional interactivity (e.g., engagement metrics)
    st.header("Engagement Metrics")
    metric_1, metric_2, metric_3 = st.columns(3)
    metric_1.metric(label="Attendance Rate", value="85%", delta="5%")
    metric_2.metric(label="Participation", value="70%", delta="2%")
    metric_3.metric(label="Homework Submission", value="90%", delta="3%")

# Run the app with: streamlit run app.py