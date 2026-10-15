"""
SmartHub Device Classification Dataset & Data Loader Module.

Handles loading, data augmentation, class weighting, and PyTorch DataLoaders
for classified device images:
  - charger (charger adapter)
  - charging_cable (charging cable)
  - power_bank (power bank)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# Ensure workspace root is always discoverable
BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

CLASS_NAMES: List[str] = ["charger", "charging_cable", "power_bank"]
HUMAN_LABELS: Dict[str, str] = {
    "charger": "Charger Adapter",
    "charging_cable": "Charging Cable",
    "power_bank": "Power Bank",
}

# Standard ImageNet normalization parameters
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(img_size: int = 224) -> Dict[str, transforms.Compose]:
    """Return train and evaluation image transformations with data augmentation."""
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    return {
        "train": train_transform,
        "valid": eval_transform,
        "test": eval_transform,
    }


class DeviceDataset(Dataset):
    """PyTorch Dataset for SmartHub device classification."""

    def __init__(
        self,
        data_dir: Union[str, Path],
        split: str = "train",
        transform: Optional[transforms.Compose] = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.split = split
        self.split_dir = self.data_dir / split
        self.classes_csv = self.split_dir / "_classes.csv"

        if not self.classes_csv.exists():
            raise FileNotFoundError(f"Classes file not found at {self.classes_csv}")

        self.df = pd.read_csv(self.classes_csv)
        self.df.columns = [c.strip() for c in self.df.columns]

        # Verify expected class columns
        for c in CLASS_NAMES:
            if c not in self.df.columns:
                raise ValueError(f"Missing expected class column '{c}' in {self.classes_csv}")

        self.transform = transform or get_transforms()[split]
        self.samples: List[Tuple[Path, int]] = []
        self.targets: List[int] = []

        for _, row in self.df.iterrows():
            filename = str(row["filename"]).strip()
            img_path = self.split_dir / filename
            if not img_path.exists():
                continue

            # Determine class index (one-hot to index)
            label_vec = [int(row[c]) for c in CLASS_NAMES]
            class_idx = int(np.argmax(label_vec))

            self.samples.append((img_path, class_idx))
            self.targets.append(class_idx)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[index]
        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def get_class_counts(self) -> Dict[str, int]:
        """Return counts of each class in the dataset."""
        counts = {name: 0 for name in CLASS_NAMES}
        for _, label in self.samples:
            counts[CLASS_NAMES[label]] += 1
        return counts

    def get_class_weights(self) -> torch.Tensor:
        """Calculate inverse frequency class weights to balance training loss."""
        counts = self.get_class_counts()
        total = len(self.samples)
        num_classes = len(CLASS_NAMES)
        weights = [total / (num_classes * max(counts[c], 1)) for c in CLASS_NAMES]
        # Normalize weights
        weights_tensor = torch.tensor(weights, dtype=torch.float32)
        weights_tensor = weights_tensor / weights_tensor.sum() * num_classes
        return weights_tensor


def get_data_loaders(
    data_dir: Union[str, Path],
    batch_size: int = 16,
    img_size: int = 224,
    num_workers: int = 0,
) -> Dict[str, DataLoader]:
    """Create DataLoaders for train, valid, and test sets."""
    transforms_dict = get_transforms(img_size=img_size)
    loaders: Dict[str, DataLoader] = {}

    for split in ["train", "valid", "test"]:
        dataset = DeviceDataset(
            data_dir=data_dir,
            split=split,
            transform=transforms_dict[split],
        )
        shuffle = split == "train"
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=False,
        )

    return loaders
