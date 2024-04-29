from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import numpy as np
import os
import Collecting

def preprocess_data():
    label_map = {label: num for num, label in enumerate(Collecting.actions)}
    sequences, labels = [], []
    for action in Collecting.actions:
        for sequence in range(Collecting.no_sequences):
            window = []
            for frame_num in range(Collecting.sequence_length):
                res = np.load(os.path.join(Collecting.DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
                window.append(res)
            sequences.append(window)
            labels.append(label_map[action])

    x = np.array(sequences)
    y = to_categorical(labels).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.05)
    return X_train, X_test, y_train, y_test
