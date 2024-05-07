# SLR algorithm based on TensorFlow

This repository contains the implementation of a sign recognition algorithm designed to accurately identify and classify signs from images. The system uses a combination of image processing and LSTM neural networks to achieve high accuracy.

**Authors:** Jevgenijs Springis and Rustams Štālbergs

**Educational institution:** Riga 80. Secondary School
>**Note:** this code was created as part of a research project (ZPD)
>
<div style="text-align: right; width: 100%;">
  <img src="/images/images(2).jpeg" width="100" height="100">
  <img src="/images/Screenshot(107).png" width="100" height="100">
  
</div>



## Project Structure

The project is structured into several Python scripts, each handling a specific part of the sign recognition process:

- `main.py`: The main driver script that integrates all components. It controls the data flow, initiates training sessions, and sets up the model evaluations.

- `Collecting.py`: Responsible for collecting image data that will be used for training the LSTM network. It includes functionalities for image capture and storage.

- `PreprocessingData.py`: Handles the preprocessing of image data to prepare it for training. This includes resizing, normalization, and augmentation techniques to enhance the dataset's variety and quality.

- `LSTM_NET.py`: Contains the implementation of the LSTM neural network used for classifying signs. This script defines the architecture, training process, and evaluation metrics.

- `MakePredictions.py`: Used to make predictions on new data using the trained LSTM model. It loads the model, processes input images, and outputs the classification results.

- `CheckCheck.py`: An additional script used for testing and verifying the accuracy and robustness of the trained model on unseen data in real time.

## Setup and Installation


These libraries were used in the code: СV2, NumPy, Matplotlib, MediaPipe, Keras, Scikit-learn.

To train a model without using your personal gestures, copy the data from the **DATA** folder and train the model, or immediately use the model prepared by **action.h5**.

# STEP 1
Using [CV2](https://opencv.org/), [Matplotlib](https://matplotlib.org/) and [NumPy](https://numpy.org/), we select points on the body with which our future model will be trained. 

<img src="/images/Screenshot(110).png" width="350" height="280">

**Facial landmarks:** 468;    **Posing landmarks:** 33;    **R-arm landmarks:** 21;    **L-hand landmarks:** 21;    **Total:** 543

# STEP 2

Acquisition and collection of data

**Collecting.py** is designed to automate the process of collecting camera data, preprocessing it, and saving it in an easy-to-use format.

<img src="/images/Screenshot(117).png" width="350" height="280">


# STEP 3
Train a deep neutral network with LSTM layers

<img src="/images/Screenshot(105).png" width="600" height="280">

The graph shows the dynamics of categorical accuracy by epoch when training a neural network. Here is a detailed description of the observed results:
## General description
**X-axis:** Shows the number of training epochs. An epoch is one pass through the entire training dataset.

**Y-axis:** Shows the categorical accuracy of the model, which measures the proportion of correctly classified examples out of all examples. The values ​​on this axis range from 0 to 1.
## Graph Analysis
- **Initial Growth:** At the beginning of training, accuracy increases rapidly, starting from about 0.75 to almost 0.9 during the first 200 epochs. This indicates that the model learns effectively from the data provided.

- **Further increase:** After 200 epochs, accuracy continues to increase, albeit at a slower rate, reaching a plateau of around 0.95 by epoch 1000.

- **Stabilization of accuracy:** From epoch 1000 to 2000, the accuracy fluctuates around 0.95-0.96, indicating the stabilization of the learning process.

- **Trends and Variability:** The orange line (mean) and light orange area (variability) show that while there is overall stabilization, there is some variation in accuracy from epoch to epoch.

## Conclusions
**Rapid improvement:** The model quickly improves its performance at the beginning of training, which is common since the initial weights are usually far from optimal.

**Plateau:** Reaching a plateau precisely indicates that further improvements to the model become less significant, perhaps due to saturation of the network's learning capacity or because all the underlying patterns in the data have already been learned.

**Possible overtraining:** The lack of significant improvement after a certain number of epochs could also indicate the onset of overtrained, especially if we had observed performance degradation on validation or test data, which is not shown here.
# STEP 4
Interactive gesture recognition application

The **CheckCheck.py** code is designed to test the trained model on a new image. This process involves loading the model, pre-processing the image, making a prediction using the model, and interpreting the prediction results to output the final result.

<div style="display: flex; justify-content: space-around;">
  <img src="/images/Screenshot(111).png" width="330" height="270">
  <img src="/images/Screenshot(112).png" width="330" height="270">
  <img src="/images/Screenshot(113).png" width="330" height="270">
</div>


# STEP 5
Improving results

- To improve the results, it was decided to record the gestures from different angles, at a different distance from the camera and in a different room.
- The number of frames that gather information was increased.
- It was decided to shorten the training periods to prevent overtraining.



