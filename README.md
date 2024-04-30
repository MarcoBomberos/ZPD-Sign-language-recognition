# SLR algorithm based on TensorFlow
**Authors:** Jevgenijs Springis and Rustams Štalbērgs
>**Note:** this code was created as part of a research project (ZPD)
>
<div style="text-align: right; width: 100%;">
  <img src="/images/Screenshot(107).png" width="100" height="100">
  <img src="/images (2).jpeg" width="100" height="100">
  
</div>

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



