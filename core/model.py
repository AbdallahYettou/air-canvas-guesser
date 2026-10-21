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


class ResidualBlock(nn.Module):
    def __init__(self, inputs, outputs, stride=1):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(inputs, outputs, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outputs), nn.ReLU(),
            nn.Conv2d(outputs, outputs, 3, padding=1, bias=False), nn.BatchNorm2d(outputs),
        )
        self.skip = nn.Identity() if inputs == outputs and stride == 1 else nn.Sequential(
            nn.Conv2d(inputs, outputs, 1, stride=stride, bias=False), nn.BatchNorm2d(outputs))
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.activation(self.body(x) + self.skip(x))


class SketchResNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 5, stride=2, padding=2, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
            ResidualBlock(32, 64, 2), ResidualBlock(64, 128, 2), ResidualBlock(128, 192, 2),
            nn.AdaptiveAvgPool2d((2, 2)),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.2), nn.Linear(192 * 4, num_classes))

    def forward(self, x):
        return self.classifier(self.features(x))


def create_model(architecture, num_classes):
    if architecture == "SketchCNN-v1":
        return SketchCNN(num_classes)
    if architecture == "SketchResNet-v1":
        return SketchResNet(num_classes)
    raise ValueError(f"Unsupported sketch architecture: {architecture}")
