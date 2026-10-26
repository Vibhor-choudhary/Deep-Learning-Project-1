"""Small, explainable neural-network models used by AksharCoach."""
import torch
from torch import nn


class MLP(nn.Module):
    """1024 -> 128 -> 46 baseline; flattening discards image geometry."""
    def __init__(self, classes: int = 46):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(), nn.Linear(32 * 32, 128), nn.ReLU(), nn.Dropout(0.20),
            nn.Linear(128, classes),
        )

    def forward(self, x):
        return self.network(x)


class CompactCNN(nn.Module):
    """1x32x32 -> 32x16x16 -> 64x8x8 -> 4096 -> 128 -> 46."""
    def __init__(self, classes: int = 46):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(64 * 8 * 8, 128), nn.ReLU(), nn.Dropout(0.30),
            nn.Linear(128, classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def build_model(name: str, classes: int = 46) -> nn.Module:
    if name.lower() == "mlp":
        return MLP(classes)
    if name.lower() == "cnn":
        return CompactCNN(classes)
    raise ValueError(f"Unknown model: {name}")


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
