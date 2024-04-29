from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import TensorBoard
import Collecting
import os
import PreprocesingData

if __name__ == "__main__":
    
    X_train, X_test, y_train, y_test = PreprocesingData.preprocess_data()

    
    log_dir = os.path.join("Logs")
    tb_callback = TensorBoard(log_dir=log_dir)

    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation="relu", input_shape=(30, 1662))) 
    model.add(LSTM(128, return_sequences=True, activation="relu"))
    model.add(LSTM(64, return_sequences=False, activation="relu"))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(Collecting.actions.shape[0], activation="softmax"))

    model.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=["categorical_accuracy"])
    model.fit(X_train, y_train, epochs=2000, callbacks=[tb_callback])

    model.save("action.h5")
