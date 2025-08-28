import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# TODO: build the neural network classifier via pytorch

# load labeled csv
data = pd.read_csv("/Users/callumanderson/Library/Mobile Documents/com~apple~CloudDocs/Documents/Documents - Callum’s Laptop/Masters-File-Repo/pytorch-learning/pricepoke/data/processed/pokemon_final_with_labels.csv")

# split features (x)
features = data[[c for c in data.columns if c != "y"]]

# split target (y)
targets = data[["y"]]

# convert the dataframe to numpy matrix to ensure shape is defined
features = features.to_numpy(dtype="float32")
targets = targets.to_numpy(dtype="float32")

# add feature matrix and target matrix to pytorch tensors
X_tensor = torch.tensor(features)
y_tensor = torch.tensor(targets)

print("loaded")