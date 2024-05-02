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

- `CheckCheck.py`: An additional script used for testing and verifying the accuracy and robustness of the trained model on unseen data.

## Setup and Installation


These libraries were used in the code: СV2, NumPy, Matplotlib, MediaPipe, Keras, Scikit-learn.


# STEP 1
Using [CV2](https://opencv.org/), [Matplotlib](https://matplotlib.org/) and [NumPy](https://numpy.org/), we select points on the body with which our future model will be trained. 
<img src="/images/Screenshot(110).png" width="350" height="280">

Sejas orientieri : 468

Pozēšanas orientieri : 33

L-rokas orientieri : 21

K-rokas orientieri : 21

Kopā: 543

# STEP 2

Datu iegūšana un vākšana

<img src="/images/Screenshot(117).png" width="350" height="280">


# STEP 3
Apmācīt dziļu neitrālu tīklu ar LSTM slāņiem sekvences noteikšanai

<img src="/images/Screenshot(105).png" width="600" height="280">


# STEP 4
Interaktīvā žestu atpazīšanas lietojumprogramma

<div style="display: flex; justify-content: space-around;">
  <img src="/images/Screenshot(111).png" width="330" height="270">
  <img src="/images/Screenshot(112).png" width="330" height="270">
  <img src="/images/Screenshot(113).png" width="330" height="270">
</div>


# STEP 5
Rezultātu uzlabošana

- Lai uzlabotu rezultātus, tika nolemts reģistrēt žestus no dažādiem leņķiem, citā attālumā no kameras un citā telpā.
- Bija palielināts to kadru skaits, kas apkopo informāciju.
- Bija nolemts saīsināt treniņu ēras, lai novērstu pārtrenēšano.



