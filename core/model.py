"""A compact CNN shared by Kaggle training and local CPU inference."""

from torch import nn


class SketchCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 24, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(24, 48, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(48, 96, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(96 * 4 * 4, 128), nn.ReLU(),
            nn.Dropout(0.25), nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))
