# 📈 Stock Price Prediction using LSTM (Netflix Inc.)

A clean, production‑ready **GitHub‑standard** README for predicting the stock price of **Netflix Inc. (NFLX)** using **LSTM networks**. Dataset used is from **Nov 2024 → Nov 2025**.

---

## 🚀 Project Overview

This project builds an end‑to‑end pipeline for stock price prediction using **Long Short-Term Memory (LSTM)** deep‑learning models.

You will:

* Preprocess Netflix stock data
* Create time‑window sequences
* Train an LSTM model
* Evaluate with RMSE
* Visualize predictions vs actual prices

---

## 📊 Dataset

* **Ticker:** NFLX (Netflix Inc.)
* **Date Range:** Nov 2024 → Nov 2025
* **Target:** Close price

Place dataset here:

```
data/netflix_2024_2025.csv
```

---

## 🧱 Project Structure

```
stock-prediction-lstm/
├── data/
│   └── netflix_2024_2025.csv
├── notebooks/
│   └── exploration.ipynb
├── src/
│   ├── preprocess.py
│   ├── model.py
│   └── train.py
├── models/
│   └── lstm_model.h5
├── results/
│   └── predictions.png
├── requirements.txt
└── README.md
```

---

## 🔧 Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/stock-prediction-lstm.git
cd stock-prediction-lstm
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install requirements

```bash
pip install -r requirements.txt
```

---

## 🧹 Data Preprocessing

Main steps:

* Handle missing values
* Scale using MinMaxScaler
* Create 60‑day window sequences

Example:

```python
scaled = scaler.fit_transform(df[['Close']])

def create_sequences(data, window):
    X, y = [], []
    for i in range(window, len(data)):
        X.append(data[i-window:i])
        y.append(data[i])
    return np.array(X), np.array(y)
```

---

## 🧠 LSTM Model Architecture

```python
model = Sequential([
    LSTM(64, input_shape=(X_train.shape[1], 1)),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')
```

Why LSTM?

* Learns long‑term patterns
* Better for financial time series

---

## 🏋️ Training

Run:

```bash
python src/train.py
```

Outputs:

* `models/lstm_model.h5`
* `results/predictions.png`

---

## 📈 Evaluation

Metric used:

* **RMSE**

Example:

```python
rmse = np.sqrt(mean_squared_error(y_test, predictions))
```

---

## 📉 Visualization

Prediction graph saved here:

```
results/predictions.png
```

Compares actual vs predicted close prices.

---

## 📦 Requirements

```
numpy
pandas
matplotlib
scikit-learn
tensorflow
keras
jupyter
```

---

## 📜 License

MIT License
