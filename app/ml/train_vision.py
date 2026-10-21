"""
SmartHub Device Classification Training & Evaluation Script.

Fine-tunes a deep vision model (MobileNetV2) using transfer learning on the
SmartHub peripheral devices dataset (powerbank, charging cable, charger adapter).
Includes class weighting, Cosine Annealing learning rate scheduling, validation
checkpointing, and holdout test set evaluation with confusion matrix generation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, f1_score

# Ensure workspace root is in sys.path
BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.ml.device_classifier import DeviceClassifier
from app.ml.device_dataset import CLASS_NAMES, HUMAN_LABELS, DeviceDataset, get_data_loaders


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """Execute one training epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / max(total, 1)
    acc = correct / max(total, 1)
    return avg_loss, acc


def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float, List[int], List[int]]:
    """Evaluate model on a validation or test set."""
    model.eval()
    total_loss = 0.0
    all_preds: List[int] = []
    all_labels: List[int] = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    total = len(all_labels)
    avg_loss = total_loss / max(total, 1)
    acc = sum(p == l for p, l in zip(all_preds, all_labels)) / max(total, 1)
    macro_f1 = float(f1_score(all_labels, all_preds, average="macro", zero_division=0))

    return avg_loss, acc, macro_f1, all_labels, all_preds


def save_confusion_matrix_plot(
    y_true: List[int],
    y_pred: List[int],
    output_path: Path,
) -> None:
    """Generate and save confusion matrix visualization."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))
    display_names = [HUMAN_LABELS[c] for c in CLASS_NAMES]

    plt.figure(figsize=(7, 6))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Device Classification Confusion Matrix (Holdout Test Set)", fontsize=13, pad=15)
    plt.colorbar()

    tick_marks = np.arange(len(CLASS_NAMES))
    plt.xticks(tick_marks, display_names, rotation=30, ha="right", fontsize=10)
    plt.yticks(tick_marks, display_names, fontsize=10)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            color = "white" if val > thresh else "black"
            plt.text(j, i, f"{val:d}", ha="center", va="center", color=color, fontsize=12, fontweight="bold")

    plt.ylabel("Ground Truth Class", fontsize=11)
    plt.xlabel("Predicted Class", fontsize=11)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_training_curves_plot(
    history: Dict[str, List[float]],
    output_path: Path,
) -> None:
    """Plot and save training vs validation loss and accuracy curves."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Loss curve
    ax1.plot(epochs, history["train_loss"], "o-", label="Train Loss", color="#1f77b4")
    ax1.plot(epochs, history["valid_loss"], "s--", label="Valid Loss", color="#ff7f0e")
    ax1.set_title("Cross-Entropy Loss vs Epoch", fontsize=12)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy curve
    ax2.plot(epochs, history["train_acc"], "o-", label="Train Acc", color="#2ca02c")
    ax2.plot(epochs, history["valid_acc"], "s--", label="Valid Acc", color="#d62728")
    ax2.set_title("Accuracy vs Epoch", fontsize=12)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def train_model(
    data_dir: str | Path = BASE_DIR / "data" / "devices",
    output_model_path: str | Path = BASE_DIR / "app" / "ml" / "device_model.pth",
    metrics_path: str | Path = BASE_DIR / "data" / "processed" / "device_vision_metrics.json",
    backbone: str = "mobilenet_v2",
    epochs: int = 15,
    batch_size: int = 16,
    lr: float = 3e-4,
    device_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Train device classifier on dataset and evaluate on test set."""
    start_time = time.time()
    data_dir = Path(data_dir)
    output_model_path = Path(output_model_path)
    metrics_path = Path(metrics_path)

    if device_name is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    print(f"==================================================")
    print(f"SmartHub Device Classifier Training Pipeline")
    print(f"==================================================")
    print(f"Device: {device}")
    print(f"Backbone: {backbone}")
    print(f"Data directory: {data_dir}")
    print(f"Epochs: {epochs}, Batch size: {batch_size}, LR: {lr}")

    loaders = get_data_loaders(data_dir, batch_size=batch_size)
    train_dataset: DeviceDataset = loaders["train"].dataset  # type: ignore

    class_weights = train_dataset.get_class_weights().to(device)
    print(f"Class distribution in train set: {train_dataset.get_class_counts()}")
    print(f"Computed loss class weights: {class_weights.cpu().numpy().round(3).tolist()}")

    # Initialize model
    model = DeviceClassifier(
        backbone=backbone,
        num_classes=len(CLASS_NAMES),
        pretrained=True,
        freeze_backbone=False,
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_f1 = -1.0
    best_epoch = 0
    history: Dict[str, List[float]] = {
        "train_loss": [],
        "train_acc": [],
        "valid_loss": [],
        "valid_acc": [],
        "valid_f1": [],
    }

    print("\n--- Starting Training ---")
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        train_loss, train_acc = train_one_epoch(model, loaders["train"], criterion, optimizer, device)
        valid_loss, valid_acc, valid_f1, _, _ = evaluate(model, loaders["valid"], criterion, device)
        scheduler.step()

        elapsed = time.time() - epoch_start
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["valid_loss"].append(valid_loss)
        history["valid_acc"].append(valid_acc)
        history["valid_f1"].append(valid_f1)

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:5.2f}% | "
            f"Valid Loss: {valid_loss:.4f} | Valid Acc: {valid_acc*100:5.2f}% | "
            f"Valid Macro-F1: {valid_f1:.4f} | Time: {elapsed:4.1f}s"
        )

        # Checkpoint if validation performance improves
        if valid_f1 > best_val_f1:
            best_val_f1 = valid_f1
            best_epoch = epoch
            model.save_checkpoint(
                output_model_path,
                optimizer=optimizer,
                epoch=epoch,
                metrics={"best_valid_acc": valid_acc, "best_valid_f1": valid_f1},
            )

    print(f"\nBest Validation Macro-F1: {best_val_f1:.4f} achieved at Epoch {best_epoch}")
    print(f"Saved best model checkpoint to: {output_model_path}")

    # Load best model for holdout test set evaluation
    best_model, _ = DeviceClassifier.load_from_checkpoint(output_model_path, device=device)
    test_loss, test_acc, test_f1, y_true, y_pred = evaluate(best_model, loaders["test"], criterion, device)

    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=[HUMAN_LABELS[c] for c in CLASS_NAMES],
        output_dict=True,
        zero_division=0,
    )
    report_text = classification_report(
        y_true,
        y_pred,
        target_names=[HUMAN_LABELS[c] for c in CLASS_NAMES],
        zero_division=0,
    )

    print("\n==================================================")
    print("Holdout Test Set Benchmark Results (30 images)")
    print("==================================================")
    print(f"Test Accuracy: {test_acc*100:.2f}%")
    print(f"Test Macro F1: {test_f1:.4f}")
    print("\nDetailed Classification Report:")
    print(report_text)

    # Save artifacts
    cm_path = BASE_DIR / "data" / "processed" / "device_confusion_matrix.png"
    save_confusion_matrix_plot(y_true, y_pred, cm_path)
    print(f"Saved confusion matrix plot: {cm_path}")

    curves_path = BASE_DIR / "data" / "processed" / "device_training_history.png"
    save_training_curves_plot(history, curves_path)
    print(f"Saved training curves plot: {curves_path}")

    total_training_time = time.time() - start_time
    summary_metrics: Dict[str, Any] = {
        "status": "success",
        "model_architecture": backbone,
        "classes": CLASS_NAMES,
        "human_labels": HUMAN_LABELS,
        "training_epochs": epochs,
        "best_epoch": best_epoch,
        "train_samples": len(train_dataset),
        "valid_samples": len(loaders["valid"].dataset),  # type: ignore
        "test_samples": len(loaders["test"].dataset),  # type: ignore
        "test_accuracy": round(test_acc, 4),
        "test_macro_f1": round(test_f1, 4),
        "classification_report": report_dict,
        "best_valid_f1": round(best_val_f1, 4),
        "total_training_time_seconds": round(total_training_time, 2),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint_path": str(output_model_path.relative_to(BASE_DIR)),
    }

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(summary_metrics, f, indent=2)

    print(f"Saved evaluation metrics JSON: {metrics_path}")
    return summary_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SmartHub Device Classifier")
    parser.add_argument("--epochs", type=int, default=12, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--backbone", type=str, default="mobilenet_v2", help="Model backbone")
    args = parser.parse_args()

    train_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        backbone=args.backbone,
    )
