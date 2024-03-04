from flask import Flask, Response, send_file, request
import cv2
import mediapipe as mp
import numpy as np
from utils import VideoProcessor, build_model, attention_block, PoseEstimationTransformer
import tempfile
import os

app = Flask(__name__)

# Load the pose estimation model from Mediapipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Create an instance of the VideoProcessor class
video_processor = VideoProcessor()

@app.route('/')
def index():
    return "Welcome to Real-time Exercise Detection"

@app.route('/process_video', methods=['POST'])
def process_video():
    video_file = request.files['video_file']
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()

    # Save the uploaded video file into the temporary directory
    uploaded_video_path = os.path.join(temp_dir, "uploaded_video.mp4")
    video_file.save(uploaded_video_path)

    # Process the video
    output_video_path = video_processor.process_video(uploaded_video_path)

    # Send the processed video as a response
    return send_file(output_video_path, as_attachment=True)

def gen_frames():
    camera = cv2.VideoCapture(0)

    while True:
        success, frame = camera.read()
        if not success:
            break

        # Process the frame using your VideoProcessor
        frame = video_processor.process(frame)

        # Encode the frame as JPEG
        _, jpeg = cv2.imencode('.jpg', frame)

        # Send the frame as a response
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + bytearray(jpeg) + b'\r\n')

    camera.release()

@app.route('/pose_estimation')
def pose_estimation():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
