import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
import json
from safetensors.torch import save_file
from dataclasses import dataclass
from typing import Tuple, List
from network import PricePredictor, get_dataloaders
import config


@dataclass
class TrainingConfig:
    batch_size: int = 64
    learning_rate: float = 0.001
    num_epochs: int = 200
    patience: int = 40
    model_save_dir: str = config.MODEL_SAVE_DIR
    input_csv_path: str = config.INPUT_CSV_PATH
    target_col: str = config.TARGET_COL


def _train_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for features, targets in dataloader:
        features, targets = features.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * features.size(0)

        predicted = torch.round(torch.sigmoid(outputs))
        total_samples += targets.size(0)
        correct_predictions += (predicted == targets).sum().item()

    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions / total_samples
    return epoch_loss, epoch_acc


def _validate_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for features, targets in dataloader:
            features, targets = features.to(device), targets.to(device)
            outputs = model(features)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * features.size(0)

            predicted = torch.round(torch.sigmoid(outputs))
            total_samples += targets.size(0)
            correct_predictions += (predicted == targets).sum().item()

    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions / total_samples
    return epoch_loss, epoch_acc


def train_model(
    model: nn.Module,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    cfg: TrainingConfig,
    feature_columns: List[str],
):
    best_val_loss = float("inf")
    input_size = len(feature_columns)
    best_val_acc = 0.0
    epochs_no_improve = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print(f"Starting training on {device} for {cfg.num_epochs} epochs...")

    for epoch in range(cfg.num_epochs):
        train_loss, train_acc = _train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = _validate_epoch(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch+1}/{cfg.num_epochs} | "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            epochs_no_improve = 0
            print(f"Validation loss decreased. Saving model to {cfg.model_save_dir}")
            os.makedirs(cfg.model_save_dir, exist_ok=True)

            config_path = os.path.join(cfg.model_save_dir, "config.json")
            model_config = {
                "input_size": input_size,
                "model_class": model.__class__.__name__,
                "feature_columns": feature_columns,
            }
            with open(config_path, "w") as f:
                json.dump(model_config, f, indent=2)

            weights_path = os.path.join(cfg.model_save_dir, "model.safetensors")
            save_file(model.state_dict(), weights_path)
            print(f"Model config and weights successfully saved to {cfg.model_save_dir}")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= cfg.patience:
            print(
                f"Early stopping triggered after {cfg.patience} epochs with no improvement."
            )
            break

    print("\nFinished Training.")
    print(
        f"Best Model Performance -> Validation Loss: {best_val_loss:.4f} | Validation Accuracy: {best_val_acc:.4f}"
    )
    print(f"Best model saved in {cfg.model_save_dir}")


def get_class_weights(train_loader: torch.utils.data.DataLoader) -> torch.Tensor:
    targets = train_loader.dataset.tensors[1]
    if isinstance(targets, torch.Tensor):
        targets = targets.numpy()

    class_counts = pd.Series(targets.flatten()).value_counts()
    count_false = class_counts.get(0, 0)
    count_true = class_counts.get(1, 0)

    pos_weight = 1.0
    if count_true > 0 and count_false > 0:
        raw_ratio = count_false / count_true
        pos_weight = np.sqrt(raw_ratio)
        print(
            f"Class Imbalance Detected in Training Set: {count_false} 'False' vs {count_true} 'True' "
            f"(Ratio: {raw_ratio:.2f})."
        )
        print(f"Applying a dampened positive class weight of {pos_weight:.2f} to compensate.")
    else:
        print("No class imbalance detected or one class is missing in the training set. Using default weight.")

    return torch.tensor([pos_weight])


if __name__ == "__main__":

    cfg = TrainingConfig()

    train_loader, val_loader, feature_columns = get_dataloaders(batch_size=cfg.batch_size)
    input_size = len(feature_columns)
    model = PricePredictor(input_size=input_size)

    pos_weight_tensor = get_class_weights(train_loader)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pos_weight_tensor = pos_weight_tensor.to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)

    train_model(model, criterion, optimizer, train_loader, val_loader, cfg, feature_columns)