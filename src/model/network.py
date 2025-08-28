import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler

input_csv_path = "/Users/callumanderson/Library/Mobile Documents/com~apple~CloudDocs/Documents/Documents - Callum’s Laptop/Masters-File-Repo/pytorch-learning/pricepoke/data/processed/pokemon_final_with_labels.csv"
encoder_path = "/Users/callumanderson/Library/Mobile Documents/com~apple~CloudDocs/Documents/Documents - Callum’s Laptop/Masters-File-Repo/pytorch-learning/pricepoke/models/encoders/scaler.pkl"

# load labeled csv
data = pd.read_csv(input_csv_path)

# split features (x)
features = data[[c for c in data.columns if c != "y"]]

# split target (y)
targets = data[["y"]]

# convert the dataframe to numpy matrix to ensure shape is defined
features = features.to_numpy(dtype="float32")
targets = targets.to_numpy(dtype="float32")

# normalize for a mean of 0 and std of 1 and save for later predictions
scaler = StandardScaler()
features = scaler.fit_transform(features)
joblib.dump(scaler, encoder_path)

# add feature matrix and target matrix to pytorch tensors
X_tensor = torch.tensor(features)
y_tensor = torch.tensor(targets)

