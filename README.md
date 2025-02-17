# Bidirectional LSTM RNN

## Introduction
This project implements a **Bidirectional LSTM (Long Short-Term Memory) Recurrent Neural Network** to process sequential data efficiently. Bidirectional LSTMs improve upon standard LSTMs by capturing dependencies in both forward and backward directions, making them useful for applications like **natural language processing (NLP), time-series forecasting, and speech recognition**.

## Features
- Implements a **Bidirectional LSTM** using TensorFlow/Keras.
- Handles sequential data with **recurrent layers**.
- Supports **training, evaluation, and inference**.
- Compatible with **GPU acceleration** for faster training.


## Usage
Run the Jupyter Notebook to train and test the model:

```sh
jupyter notebook Bidirectional_LSTM_RNN.ipynb
```

### Training
Inside the notebook, configure the hyperparameters and run the training cell to train the model on your dataset.

### Evaluation
Once training is complete, evaluate the model performance using test data and visualize the results.

### Inference
Use the trained model to make predictions on new input sequences.

## Dependencies
Ensure the following libraries are installed:
- TensorFlow/Keras
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Jupyter Notebook

Install missing dependencies using:

```sh
pip install tensorflow numpy pandas matplotlib scikit-learn jupyter
```
