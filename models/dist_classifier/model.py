import torch
from torchvision.models import ResNet18_Weights, resnet18

class DistortionClassifier(torch.nn.Module):
    """ImageNet-pretrained ResNet-18 distortion classifier."""

    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = resnet18(weights=weights)
        self.backbone.fc = torch.nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.backbone(images)
