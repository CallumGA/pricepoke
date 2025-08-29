import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
import config


"""
    Prepare the data for training
"""


def get_dataloaders(batch_size=64):

    # load labeled csv
    data = pd.read_csv(config.INPUT_CSV_PATH)

    # define features vs. targets in the config, pull them for use in network
    feature_cols = [c for c in data.columns if c not in config.IDENTIFIER_COLS and c != config.TARGET_COL]

    if not feature_cols:
        raise ValueError("No feature columns were found. Check your CSV and identifier_cols list.")

    # split features (x)
    features = data[feature_cols]

    # split target (y)
    targets = data[[config.TARGET_COL]]

    # convert the dataframe to numpy matrix to ensure shape is defined
    features = features.to_numpy(dtype="float32")
    targets = targets.to_numpy(dtype="float32")

    # normalize for a mean of 0 and std of 1 and save pk1 for later predictions
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    joblib.dump(scaler, config.SCALER_PATH)

    # now we must do a train/test split for features (x) and for targets (y)
    feature_train, feature_val, target_train, target_val = train_test_split(
        features, targets, test_size=0.2, random_state=42
    )

    # convert the split numpy arrays to PyTorch tensors and create datasets
    train_dataset = TensorDataset(torch.tensor(feature_train), torch.tensor(target_train))
    val_dataset = TensorDataset(torch.tensor(feature_val), torch.tensor(target_val))

    # create dataloaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, feature_train.shape[1]


"""
    Neural Network Classifier Architecture
"""
class PricePredictor(nn.Module):
    def __init__(self, input_size):
        super(PricePredictor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 256),    # <-- input layer
            nn.ReLU(),
            nn.Linear(256, 128), # <-- hidden layer
            nn.ReLU(),
            nn.Linear(128, 1)    # <-- output layer
        )

    def forward(self, x):
        return self.model(x)
