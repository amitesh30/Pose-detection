import os
import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import math
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (LSTM, Dense, Dropout, Input, Flatten, 
                                     Bidirectional, Permute, multiply)
import mediapy
import tempfile
import shutil


# Load the pose estimation model from Mediapipe
mp_pose = mp.solutions.pose 
mp_drawing = mp.solutions.drawing_utils 
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) 

# Define the attention block for the LSTM model
def attention_block(inputs, time_steps):
    a = Permute((2, 1))(inputs)
    a = Dense(time_steps, activation='softmax')(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    output_attention_mul = multiply([inputs, a_probs], name='attention_mul') 
    return output_attention_mul

# Build and load the LSTM model
@st.cache(allow_output_mutation=True)
def build_model(HIDDEN_UNITS=256, sequence_length=30, num_input_values=33*4, num_classes=3):
    inputs = Input(shape=(sequence_length, num_input_values))
    lstm_out = Bidirectional(LSTM(HIDDEN_UNITS, return_sequences=True))(inputs)
    attention_mul = attention_block(lstm_out, sequence_length)
    attention_mul = Flatten()(attention_mul)
    x = Dense(2*HIDDEN_UNITS, activation='relu')(attention_mul)
    x = Dropout(0.5)(x)
    x = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=[inputs], outputs=x)
    load_dir = "LSTM_Attention.h5"  
    model.load_weights(load_dir)
    return model

# Define the VideoProcessor class for real-time video processing
class VideoProcessor:
    def __init__(self):
        # Parameters
        self.actions = np.array(['curl', 'press', 'squat'])
        self.sequence_length = 30
        self.colors = [(245,117,16), (117,245,16), (16,117,245)]
        self.threshold = 0.5

        self.model = build_model(256)
        
        # Detection variables
        self.sequence = []
        self.current_action = ''

        # Rep counter logic variables
        self.curl_counter = 0
        self.press_counter = 0
        self.squat_counter = 0
        self.curl_stage = None
        self.press_stage = None
        self.squat_stage = None
        self.pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

  
    def process_video(self, video_path, output_dir):
        # Create a temporary directory within the output directory
        temp_output_dir = tempfile.mkdtemp(dir=output_dir)

        # Process the video and save the processed video to the temporary output directory
        output_filename = os.path.join(temp_output_dir, "processed_video.mp4")
        cap = cv2.VideoCapture(video_path)
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        out = cv2.VideoWriter(output_filename, cv2.VideoWriter_fourcc(*'h264'), 30, (frame_width, frame_height))
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(frame_rgb)
            processed_frame = self.process_frame(frame, results)
            out.write(processed_frame)
        cap.release()
        out.release()

        # Return the path to the processed video file
        return output_filename
    
    def process_frame(self, frame, results):
        # Process the frame using the `process` function
        processed_frame = self.process(frame)
        return processed_frame
    
    def process(self, image):
      
        # Pose detection model
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        self.draw_landmarks(image, results) 
        
        # Prediction logic
        keypoints = self.extract_keypoints(results)        
        self.sequence.append(keypoints.astype('float32',casting='same_kind'))      
        self.sequence = self.sequence[-self.sequence_length:]
        
        if len(self.sequence) == self.sequence_length:
            res = self.model.predict(np.expand_dims(self.sequence, axis=0), verbose=0)[0]
            
            self.current_action = self.actions[np.argmax(res)]
            confidence = np.max(res)
            print("confidence", confidence)  # Debug print statement
            print("current action" , self.current_action)
            
            # Erase current action variable if no probability is above threshold
            if confidence < self.threshold:
                self.current_action = ''
                
                
            print("current action" , self.current_action)


            # Viz probabilities
            image = self.prob_viz(res, image)
            
            # Count reps
            
            landmarks = results.pose_landmarks.landmark
            self.count_reps(image, landmarks, mp_pose)
            

            # Display graphical information
            cv2.rectangle(image, (0,0), (640, 40), self.colors[np.argmax(res)], -1)
            cv2.putText(image, 'curl ' + str(self.curl_counter), (3,30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, 'press ' + str(self.press_counter), (240,30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, 'squat ' + str(self.squat_counter), (490,30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
          
        return image

    def draw_landmarks(self, image, results):
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                  mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))
        return image

    def extract_keypoints(self, results):
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
        return pose

    def count_reps(self, image, landmarks, mp_pose):
        """
        Counts repetitions of each exercise. Global count and stage (i.e., state) variables are updated within this function.

        """

        if self.current_action == 'curl':
            # Get coords
            shoulder = self.get_coordinates(landmarks, mp_pose, 'LEFT', 'SHOULDER')
            elbow = self.get_coordinates(landmarks, mp_pose, 'LEFT', 'ELBOW')
            wrist = self.get_coordinates(landmarks, mp_pose, 'LEFT', 'WRIST')

            # calculate elbow angle
            angle = self.calculate_angle(shoulder, elbow, wrist)

            # curl counter logic
            print("Curl Angle:", angle)  # Debug print statement
            if angle < 30:
                self.curl_stage = "up"
            if angle > 140 and self.curl_stage == 'up':
                self.curl_stage = "down"
                self.curl_counter += 1
                print("count:",self.curl_counter)
            self.press_stage = None
            self.squat_stage = None

            # Viz joint angle
            self.viz_joint_angle(image, angle, elbow)

        elif self.current_action == 'press':
            # Get coords
            shoulder = self.get_coordinates(landmarks, mp_pose, 'LEFT', 'SHOULDER')
            elbow = self.get_coordinates(landmarks, mp_pose, 'LEFT', 'ELBOW')
            wrist = self.get_coordinates(landmarks, mp_pose, 'LEFT', 'WRIST')

            # Calculate elbow angle
            elbow_angle = self.calculate_angle(shoulder, elbow, wrist)
            print(shoulder, elbow, wrist)
            # Compute distances between joints
            shoulder2elbow_dist = abs(math.dist(shoulder, elbow))
            shoulder2wrist_dist = abs(math.dist(shoulder, wrist))

            # Press counter logic
            print("Press Angle:", elbow_angle)  # Debug print statement
            print("Shoulder to Elbow Distance:", shoulder2elbow_dist)  # Debug print statement
            print("Shoulder to Wrist Distance:", shoulder2wrist_dist)  # Debug print statement
            if (elbow_angle > 130) and (shoulder2elbow_dist < shoulder2wrist_dist):
                self.press_stage = "up"
            if (elbow_angle < 50) and (shoulder2elbow_dist > shoulder2wrist_dist) and (self.press_stage == 'up'):
                self.press_stage = 'down'
                self.press_counter += 1
                
                
                print("count:",self.press_counter)
            self.curl_stage = None
            self.squat_stage = None

            # Viz joint angle
            self.viz_joint_angle(image, elbow_angle, elbow)

        elif self.current_action == 'squat':
            # Get coords
            # left side
            left_shoulder = self.get_coordinates(landmarks, mp_pose, 'LEFT', 'SHOULDER')
            left_hip = self.get_coordinates(landmarks, mp_pose, 'LEFT', 'HIP')
            left_knee = self.get_coordinates(landmarks, mp_pose, 'LEFT', 'KNEE')
            left_ankle = self.get_coordinates(landmarks, mp_pose, 'LEFT', 'ANKLE')
            # right side
            right_shoulder = self.get_coordinates(landmarks, mp_pose, 'RIGHT', 'SHOULDER')
            right_hip = self.get_coordinates(landmarks, mp_pose, 'RIGHT', 'HIP')
            right_knee = self.get_coordinates(landmarks, mp_pose, 'RIGHT', 'KNEE')
            right_ankle = self.get_coordinates(landmarks, mp_pose, 'RIGHT', 'ANKLE')

            # Calculate knee angles
            left_knee_angle = self.calculate_angle(left_hip, left_knee, left_ankle)
            right_knee_angle = self.calculate_angle(right_hip, right_knee, right_ankle)

            # Calculate hip angles
            left_hip_angle = self.calculate_angle(left_shoulder, left_hip, left_knee)
            right_hip_angle = self.calculate_angle(right_shoulder, right_hip, right_knee)

            # Squat counter logic
            thr = 165
            print("Left Knee Angle:", left_knee_angle)  # Debug print statement
            print("Right Knee Angle:", right_knee_angle)  # Debug print statement
            print("Left Hip Angle:", left_hip_angle)  # Debug print statement
            print("Right Hip Angle:", right_hip_angle)  # Debug print statement
            if (left_knee_angle < thr) and (right_knee_angle < thr) and (left_hip_angle < thr) and (
                    right_hip_angle < thr):
                self.squat_stage = "down"
            if (left_knee_angle > thr) and (right_knee_angle > thr) and (left_hip_angle > thr) and (
                    right_hip_angle > thr) and (self.squat_stage == 'down'):
                self.squat_stage = 'up'
                self.squat_counter += 1
                print("count:",self.squat_counter)
            self.curl_stage = None
            self.press_stage = None

            # Viz joint angles
            self.viz_joint_angle(image, left_knee_angle, left_knee)
            self.viz_joint_angle(image, left_hip_angle, left_hip)

        else:
            pass
        return

    def prob_viz(self, res, input_frame):
        """
        This function displays the model prediction probability distribution over the set of exercise classes
        as a horizontal bar graph
        
        """
        output_frame = input_frame.copy()
        for num, prob in enumerate(res):        
            cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), self.colors[num], -1)
            cv2.putText(output_frame, self.actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            
        return output_frame

    def get_coordinates(self, landmarks, mp_pose, side, part):
        
        
        coord = getattr(mp_pose.PoseLandmark,side.upper()+"_"+part.upper())
        x_coord_val = landmarks[coord.value].x
        y_coord_val = landmarks[coord.value].y
        return [x_coord_val, y_coord_val] 
    
    
   
       



    def calculate_angle(self, a, b, c):
        a = np.array(a) 
        b = np.array(b) 
        c = np.array(c)
        radians = math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)
        if angle > 180.0:
            angle = 360 - angle
        return angle

    def viz_joint_angle(self, image, angle, joint):
        cv2.putText(image, str(round(angle, 2)), 
                    tuple(np.multiply(joint, [640, 480]).astype(int)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)

# Define Streamlit app

def main():
    st.title("Real-time Exercise Detection")
    video_file = st.file_uploader("Upload a video file", type=["mp4", "avi"])
    if video_file is not None:
        # Create a temporary directory
        temp_dir = tempfile.mkdtemp()

        # Save the uploaded video file into the temporary directory
        uploaded_video_path = os.path.join(temp_dir, "uploaded_video.mp4")
        with open(uploaded_video_path, 'wb') as temp_file:
            temp_file.write(video_file.read())

        # Process the video
        video_processor = VideoProcessor()
        output_dir = "output"  # Output directory
        os.makedirs(output_dir, exist_ok=True)  # Create the output directory if it doesn't exist
        output_video_path = video_processor.process_video(uploaded_video_path, output_dir)

        # Display the processed video
      
        st.video(output_video_path)

        # Remove the temporary directory and its contents
        

if __name__ == "__main__":
    main()
