###  1. Import and Instal Dependencies  ###

# pip install tensorflow tensorflow-gpu opencv-python mediapipe sklearn matplotlib
import cv2                                 # to use our webcam to collect datas
import numpy as np                         # is going to help us working with different arrays later on and how we actually structure our different data sets
import os                                  # to make it easier to work with file paths
from matplotlib import pyplot as plt       # make it easier to visualize images 
import time                                # to take a sleep between each frame that we collect, gives us time to et into position
import mediapipe as mp                     # to extract keypoints values

### 2. Keypoints using MP Holistic ###

mp_holistic = mp.solutions.holistic        # Holistic model
mp_drawing = mp.solutions.drawing_utils    # Drawing utilities

def mediapipe_detection(image, model):
  image  = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR TO RGB  "cvtColor - converts an input image from one color space to another"
  image.flags.writeable = False                   # Image is no longer writeable 
  results = model.process(image)                  # make predition
  image.flags.writeable = True                    # image is now writeable
  image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # COLOR CONVERSION RGB TO BGR
  return image, results

def draw_landmarks(image, results):
   mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS) # Draw face connections "mp_drawing.draw_landmarks make it easier to draw a landmarks"
   mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS) # Draw pose connections
   mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw left hand connections
   mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw right hand connections

def draw_styled_landmarks(image, results):
    # Draw face connections
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                              mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                              mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
                              )
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
                              )

    # Draw left-hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
                              )

    # Draw right-hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                              )
if __name__ == "__main__":
    # This part of the code will only run when main.py is executed directly, not when it's imported as a module.
    cap = cv2.VideoCapture(0)  # accessing our webcam
    # Set mediapipe model
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            # Read feed
            ret, frame = cap.read()
            # Make a detections
            image, results = mediapipe_detection(frame, holistic)
            print(results)
            # Draw Landmarks
            draw_styled_landmarks(image, results)
            # Show to screen
            cv2.imshow("OpenCv Feed", image)
            # Breaking gracefully from the loop
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

### 3. Extract Keypoint Values ###

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() \
        if results.pose_landmarks.landmark else np.zeros(33*4)

    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() \
        if results.face_landmarks else np.zeros(468 * 3)

    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() \
        if results.left_hand_landmarks else np.zeros(21 * 3)

    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() \
        if results.right_hand_landmarks else np.zeros(21 * 3)
    return np.concatenate([pose, face, lh, rh])
