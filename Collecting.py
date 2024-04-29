import os
import numpy as np
import cv2
from main import mp_holistic, mediapipe_detection, draw_styled_landmarks, extract_keypoints

# Path for exported data numpy arrays
DATA_PATH = os.path.join("MP_Data2")

# Actions that we try to detect
actions = np.array(["hello", "iloveyou", "thanks"])

# Thirty videos worth of data
no_sequences = 30

# Videos are going to be 30 frames in length
sequence_length = 30

for action in actions:
    for sequence in range(no_sequences):
        try:
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            pass

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    # Set mediapipe model
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        # NEW LOOP
        # Loop through actions
        for action in actions:
            # Loop through sequences aka videos
            for sequence in range(no_sequences):
                # Loop through video length aka sequence length
                for frame_num in range(sequence_length):
                    # Read feed
                    ret, frame = cap.read()
                    # Make detections
                    image, results = mediapipe_detection(frame, holistic)
                    print(results)
                    # Draw landmarks
                    draw_styled_landmarks(image, results)
                    # Apply wait Logic
                    if frame_num == 0:
                        cv2.putText(image, "STARTING COLLECTION", (120, 200),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                        cv2.putText(image, "Collection frames for {} Video Number {}".format(action, sequence), (15, 12),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        cv2.waitKey(2000)
                    else:
                        cv2.putText(image, "Collecting frames for {} Video Number {}".format(action, sequence), (15, 12),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    # NEW Export keypoints
                    keypoints = extract_keypoints(results)
                    npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                    np.save(npy_path, keypoints)
                    # Show to screen
                    cv2.imshow("OpenCv Feed", image)
                    # Break gracefully
                    if cv2.waitKey(10) & 0xFF == ord("q"):
                        break
        cap.release()
        cv2.destroyAllWindows()
