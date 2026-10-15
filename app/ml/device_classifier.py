"""
SmartHub Device Classifier Model Architecture.

Implements deep convolutional neural networks with transfer learning (MobileNetV2 / ResNet18)
specifically optimized for classifying SmartHub devices:
  - 0: charger (Charger Adapter)
  - 1: charging_cable (Charging Cable)
  - 2: power_bank (Power Bank)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Ensure workspace root is always discoverable for imports
BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import torch
import torch.nn as nn
from torchvision import models

try:
    from app.ml.device_dataset import CLASS_NAMES, HUMAN_LABELS
except ImportError:
    CLASS_NAMES: List[str] = ["charger", "charging_cable", "power_bank"]
    HUMAN_LABELS: Dict[str, str] = {
        "charger": "Charger Adapter",
        "charging_cable": "Charging Cable",
        "power_bank": "Power Bank",
    }


class DeviceClassifier(nn.Module):
    """Deep learning classifier for SmartHub peripheral devices."""

    def __init__(
        self,
        backbone: str = "mobilenet_v2",
        num_classes: int = len(CLASS_NAMES),
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout_rate: float = 0.3,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone
        self.num_classes = num_classes
        self.class_names = CLASS_NAMES
        self.human_labels = HUMAN_LABELS
        self.base_model: nn.Module

        if backbone == "mobilenet_v2":
            weights = (
                models.MobileNet_V2_Weights.DEFAULT
                if (pretrained and hasattr(models, "MobileNet_V2_Weights"))
                else None
            )
            mobilenet: Any = models.mobilenet_v2(weights=weights)

            if freeze_backbone and hasattr(mobilenet, "features"):
                for param in mobilenet.features.parameters():
                    param.requires_grad = False

            # Extract input feature count from existing classification head (defaults to 1280)
            in_features = 1280
            if hasattr(mobilenet, "classifier") and len(mobilenet.classifier) > 1:
                last_layer: Any = mobilenet.classifier[1]
                in_features = getattr(last_layer, "in_features", 1280)

            # Custom classification head
            mobilenet.classifier = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(in_features, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_rate * 0.7),
                nn.Linear(256, num_classes),
            )
            self.base_model = mobilenet

        elif backbone == "resnet18":
            weights = (
                models.ResNet18_Weights.DEFAULT
                if (pretrained and hasattr(models, "ResNet18_Weights"))
                else None
            )
            resnet: Any = models.resnet18(weights=weights)

            if freeze_backbone:
                for name, param in resnet.named_parameters():
                    if "fc" not in name:
                        param.requires_grad = False

            in_features = getattr(resnet.fc, "in_features", 512)
            resnet.fc = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(in_features, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_rate * 0.7),
                nn.Linear(256, num_classes),
            )
            self.base_model = resnet

        else:
            raise ValueError(f"Unsupported backbone: {backbone}. Choose 'mobilenet_v2' or 'resnet18'.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning unnormalized class logits."""
        result: torch.Tensor = self.base_model(x)  # type: ignore[operator]
        return result

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning softmax probability distributions."""
        logits = self.forward(x)
        return torch.softmax(logits, dim=-1)

    def save_checkpoint(
        self,
        filepath: Union[str, Path],
        optimizer: Optional[torch.optim.Optimizer] = None,
        epoch: int = 0,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Save full model checkpoint including architecture configuration and weights."""
        save_path = Path(filepath)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "backbone": self.backbone_name,
            "num_classes": self.num_classes,
            "class_names": self.class_names,
            "human_labels": self.human_labels,
            "state_dict": self.state_dict(),
            "epoch": epoch,
            "metrics": metrics or {},
        }
        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        torch.save(checkpoint, save_path)

    @classmethod
    def load_from_checkpoint(
        cls,
        filepath: Union[str, Path],
        device: Optional[torch.device] = None,
    ) -> Tuple[DeviceClassifier, Dict[str, Any]]:
        """Load trained model instance and metadata from checkpoint."""
        load_path = Path(filepath)
        if not load_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found at: {load_path}")

        target_device = device or torch.device("cpu")
        checkpoint = torch.load(load_path, map_location=target_device, weights_only=False)
        backbone = checkpoint.get("backbone", "mobilenet_v2")
        num_classes = checkpoint.get("num_classes", len(CLASS_NAMES))

        model = cls(backbone=backbone, num_classes=num_classes, pretrained=False)
        model.load_state_dict(checkpoint["state_dict"])
        model.to(target_device)
        model.eval()

        metadata = {
            "epoch": checkpoint.get("epoch", 0),
            "metrics": checkpoint.get("metrics", {}),
            "class_names": checkpoint.get("class_names", CLASS_NAMES),
            "human_labels": checkpoint.get("human_labels", HUMAN_LABELS),
        }
        return model, metadata
