import os
import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import math

from streamlit_webrtc import webrtc_streamer
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (LSTM, Dense, Dropout, Input, Flatten, 
                                     Bidirectional, Permute, multiply)
import mediapy
import tempfile
import shutil

from utils import VideoProcessor, build_model, attention_block, PoseEstimationTransformer

# Load the pose estimation model from Mediapipe
mp_pose = mp.solutions.pose 
mp_drawing = mp.solutions.drawing_utils 
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) 


# Define Streamlit app
def main():
    st.title("Real-time Exercise Detection")
    # video_file = st.file_uploader("Upload a video file", type=["mp4", "avi"])
    # if video_file is not None:
    #     # Create a temporary directory
    #     temp_dir = tempfile.mkdtemp()
    #
    #     # Save the uploaded video file into the temporary directory
    #     uploaded_video_path = os.path.join(temp_dir, "uploaded_video.mp4")
    #     with open(uploaded_video_path, 'wb') as temp_file:
    #         temp_file.write(video_file.read())
    #
    #     # Process the video
    #     video_processor = VideoProcessor()
    #     output_video_path = video_processor.process_video(uploaded_video_path)
    #
    #     # Display the processed video
    #     st.video(output_video_path)
    #
    #     # Create a download button for the processed video
    #     with open(output_video_path, "rb") as video_file:
    #         video_bytes = video_file.read()
    #         st.download_button(
    #             label="Download Processed Video",
    #             data=video_bytes,
    #             file_name="processed_video.mp4",
    #             mime="video/mp4"
    #         )
    #
    #     # Remove the temporary directory and its contents
    #     shutil.rmtree(temp_dir)
    #
    #

    st.title("Real-time Exercise Detection")

    # Create an instance of the VideoProcessor class
    video_processor = VideoProcessor()

    # Process webcam stream
    process_webcam(video_processor.process)


def process_webcam(process_fn):
    st.title("Pose Estimation using Webcam")

    # Initialize pose estimation transformer
    pose_transformer = PoseEstimationTransformer(process_fn)

    # Start the webcam stream
    webrtc_streamer(key="poseEstimation", video_processor_factory=pose_transformer)
        

if __name__ == "__main__":
    main()
