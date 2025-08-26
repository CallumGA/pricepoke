import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# TODO: build the neural network classifier via pytorch

# load labeled csv
data = pd.read_csv("/Users/callumanderson/Documents/Documents - Callum’s Laptop/Masters-File-Repo/pytorch-learning/pricepoke/data/processed/pokemon_final_with_labels.csv")

# split features (x)
features = data[[c for c in data.columns if c != "y"]]

# split target (y)
targets = data[["y"]]

# reshape tensors?
