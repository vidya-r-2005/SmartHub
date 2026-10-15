"""
Unit tests for SmartHub Device Classification and Vision Verification.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
import torch
from PIL import Image

from app.ml.device_classifier import DeviceClassifier
from app.ml.device_dataset import CLASS_NAMES, HUMAN_LABELS, DeviceDataset, get_transforms
from app.ml.vision import ITEM_ALIASES, DeviceVerifier, verify_item



class TestDeviceDataset(unittest.TestCase):
    """Tests for dataset loading, transforms, and class weight calculations."""

    @classmethod
    def setUpClass(cls):
        cls.data_dir = Path("data/devices")

    def test_class_constants(self):
        self.assertEqual(len(CLASS_NAMES), 3)
        self.assertIn("charger", CLASS_NAMES)
        self.assertIn("charging_cable", CLASS_NAMES)
        self.assertIn("power_bank", CLASS_NAMES)

    def test_transforms_shape(self):
        t_dict = get_transforms(img_size=224)
        dummy_img = Image.new("RGB", (300, 300), color=(100, 150, 200))

        train_tensor = t_dict["train"](dummy_img)
        self.assertEqual(train_tensor.shape, torch.Size([3, 224, 224]))

        eval_tensor = t_dict["valid"](dummy_img)
        self.assertEqual(eval_tensor.shape, torch.Size([3, 224, 224]))

    def test_dataset_loading_if_exists(self):
        if self.data_dir.exists() and (self.data_dir / "train" / "_classes.csv").exists():
            ds = DeviceDataset(self.data_dir, split="train")
            self.assertGreater(len(ds), 0)
            img, label = ds[0]
            self.assertEqual(img.shape, torch.Size([3, 224, 224]))
            self.assertIn(label, [0, 1, 2])

            weights = ds.get_class_weights()
            self.assertEqual(weights.shape[0], 3)
            self.assertTrue(torch.all(weights > 0))


class TestDeviceClassifierModel(unittest.TestCase):
    """Tests for model architecture, forward pass, and checkpointing."""

    def test_forward_pass_mobilenet(self):
        model = DeviceClassifier(backbone="mobilenet_v2", num_classes=3, pretrained=False)
        dummy_input = torch.randn(2, 3, 224, 224)
        logits = model(dummy_input)
        self.assertEqual(logits.shape, torch.Size([2, 3]))

        probs = model.predict_proba(dummy_input)
        self.assertEqual(probs.shape, torch.Size([2, 3]))
        np.testing.assert_allclose(probs.sum(dim=-1).detach().numpy(), np.array([1.0, 1.0]), atol=1e-5)

    def test_checkpoint_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "test_model.pth"
            model = DeviceClassifier(backbone="mobilenet_v2", num_classes=3, pretrained=False)
            model.save_checkpoint(ckpt_path, epoch=5, metrics={"acc": 0.95})

            self.assertTrue(ckpt_path.exists())
            loaded_model, meta = DeviceClassifier.load_from_checkpoint(ckpt_path)
            self.assertEqual(meta["epoch"], 5)
            self.assertEqual(meta["metrics"]["acc"], 0.95)

            dummy_input = torch.randn(1, 3, 224, 224)
            out = loaded_model(dummy_input)
            self.assertEqual(out.shape, torch.Size([1, 3]))


class TestDeviceVerifierAPI(unittest.TestCase):
    """Tests for verification API and alias resolution."""

    def test_item_aliases(self):
        self.assertEqual(ITEM_ALIASES.get("powerbank"), "power_bank")
        self.assertEqual(ITEM_ALIASES.get("cable"), "charging_cable")
        self.assertEqual(ITEM_ALIASES.get("adapter"), "charger")
        self.assertEqual(ITEM_ALIASES.get("charger_adapter"), "charger")

    def test_nonexistent_image_raises_error(self):
        non_existent = Path("data/nonexistent_image_12345.jpg")
        with self.assertRaises(FileNotFoundError):
            verify_item(non_existent)


if __name__ == "__main__":
    unittest.main()
