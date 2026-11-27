import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, models

df = pd.read_csv("data/H17K-1_0.2m_daily.csv")

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

# Extract temperature values
temperatures = df["temp_c"].values.reshape(-1, 1)

### TODO: CREATE TRAINING SPLITS 

# CNN parameters
n_filters = 32
filter_width = 2
dilation_rates = [2**i for i in range(8)]

# Forecasting parameters
lookback = 180 # ~6-months look back, feel free to experiement with whatever you like
horizon = 14 # forecast two weeks 

model = models.Sequential([
    # Input layer
    layers.Input(shape=(None, 1)),

    # Dilated causal convolutional layers
    layers.Conv1D(
        filters=n_filters,
        kernel_size=filter_width,
        dilation_rate=1,
        padding="causal",
        activation="relu"
    ),

    layers.Conv1D(
        filters=n_filters,
        kernel_size=filter_width,
        dilation_rate=2,
        padding="causal",
        activation="relu"
    ),

    layers.Conv1D(
        filters=n_filters,
        kernel_size=filter_width,
        dilation_rate=4,
        padding="causal",
        activation="relu"
    ),

    layers.Conv1D(
        filters=n_filters,
        kernel_size=filter_width,
        dilation_rate=8,
        padding="causal",
        activation="relu"
    ),

    layers.Conv1D(
        filters=n_filters,
        kernel_size=filter_width,
        dilation_rate=16,
        padding="causal",
        activation="relu"
    ),

    layers.Conv1D(
        filters=n_filters,
        kernel_size=filter_width,
        dilation_rate=32,
        padding="causal",
        activation="relu"
    ),

    layers.Conv1D(
        filters=n_filters,
        kernel_size=filter_width,
        dilation_rate=64,
        padding="causal",
        activation="relu"
    ),

    layers.Conv1D(
        filters=n_filters,
        kernel_size=filter_width,
        dilation_rate=128,
        padding="causal",
        activation="relu"
    ),

    # Fully connected layers
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.2),
    layers.Dense(1),

    # Extract the final 14 predictions
    layers.Lambda(lambda x: x[:, -horizon:, :])
])

model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)

model.summary()

### TODO: train + evaluate
# Feel free to change the parameters or the number of layers if you would like