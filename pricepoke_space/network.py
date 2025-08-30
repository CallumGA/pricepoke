import torch
import torch.nn as nn

"""
    Neural Network Classifier Architecture
"""

class PricePredictor(nn.Module):

    def __init__(self, input_size: int):
        super(PricePredictor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)