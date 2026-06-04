# Sentiment Analysis on IMDB Reviews using LSTM

A deep learning model that classifies IMDB movie reviews as positive or negative using a Long Short-Term Memory (LSTM) network.

---

## Overview

- Preprocesses and tokenizes 50,000 IMDB movie reviews
- Builds an LSTM model with an Embedding layer, LSTM layers, and Dense output
- Trained on the Keras IMDB dataset (25k train / 25k test)
- Achieves ~88% test accuracy

## Tech Stack

Python, TensorFlow / Keras, NumPy, Matplotlib

## Architecture

```
Input → Embedding(10000, 128) → LSTM(128) → Dropout(0.5) → Dense(1, sigmoid)
```

## Run

```bash
pip install -r requirements.txt
jupyter notebook sentiment_analysis_lstm.ipynb
```
