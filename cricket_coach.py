import streamlit as st
import openai
import cv2
import mediapipe as mp
import tempfile
import os
from moviepy.editor import VideoFileClip
import numpy as np

# Set OpenAI API Key (Use your own API Key)
OPENAI_API_KEY = "your-api-key-here"

openai.api_key = OPENAI_API_KEY

# Streamlit App Title
st.set_page_config(page_title="AI Cricketing Coach", layout="wide")
st.title("🏏 AI Cricketing Coach - Chat & Video Analysis")

# Sidebar Navigation
st.sidebar.header("Navigation")
option = st.sidebar.radio("Choose an option:", ["🏏 Chat-based Coaching", "📹 Video Analysis"])

# Function to get AI response from GPT-4o
def get_ai_response(user_query):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": "You are an expert cricket coach providing insights on batting, bowling, fitness, nutrition, and mindset."},
                      {"role": "user", "content": user_query}]
        )
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error: {e}"

# **🏏 Chat-based Coaching**
if option == "🏏 Chat-based Coaching":
    st.subheader("💬 Chat with the AI Cricket Coach")
    user_input = st.text_area("Ask me anything about cricket training, technique, fitness, or mindset!")

    if st.button("Get Coaching Advice"):
        if user_input:
            response = get_ai_response(user_input)
            st.success(response)
        else:
            st.warning("Please enter a question.")

# **📹 Video Analysis - Cricket Batting/Bowling Form**
elif option == "📹 Video Analysis":
    st.subheader("📹 Upload your cricket video for AI-based analysis")
    uploaded_file = st.file_uploader("Upload a video (MP4, MOV, AVI)", type=["mp4", "mov", "avi"])

    if uploaded_file is not None:
        # Save the uploaded video to a temporary file
        temp_dir = tempfile.mkdtemp()
        video_path = os.path.join(temp_dir, uploaded_file.name)
        
        with open(video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Display uploaded video
        st.video(video_path)

        # Load video for frame-by-frame analysis
        st.subheader("⚡ AI Analysis of Your Cricket Technique")

        # Load MediaPipe Pose model
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose()

        # Open video with OpenCV
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        key_frames = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            if frame_count % 10 == 0:  # Analyze every 10th frame for efficiency
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb_frame)

                if results.pose_landmarks:
                    annotated_frame = frame.copy()
                    mp.solutions.drawing_utils.draw_landmarks(annotated_frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                    # Convert frame for Streamlit display
                    key_frames.append(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))

        cap.release()
        pose.close()

        # Display key analysis frames
        if key_frames:
            st.subheader("📌 Key AI Analyzed Frames")
            st.image(key_frames, caption="Batting/Bowling Stance Analysis", use_column_width=True)
            st.success("✅ AI has analyzed your technique! Consult a professional coach for deeper insights.")

        else:
            st.warning("No keyframes detected. Try uploading a clearer video.")

st.sidebar.info("Built using **GPT-4o, MediaPipe & OpenAI API** 🚀")
