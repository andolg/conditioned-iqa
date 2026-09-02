import torch
from torchvision.models import ResNet18_Weights, resnet18


class DistortionClassifier(torch.nn.Module):
    """ImageNet-pretrained ResNet-18 distortion classifier."""

    FEATURE_DIMS = {"layer1": 64, "layer2": 128, "layer3": 256, "layer4": 512}

    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = resnet18(weights=weights)
        self.backbone.fc = torch.nn.Linear(self.backbone.fc.in_features, num_classes)

    @classmethod
    def feature_dim(cls, layer: str) -> int:
        try:
            return cls.FEATURE_DIMS[layer]
        except KeyError as error:
            choices = ", ".join(cls.FEATURE_DIMS)
            raise ValueError(
                f"unknown classifier feature layer {layer!r}; choose one of {choices}"
            ) from error

    def extract_features(self, images: torch.Tensor, layer: str) -> torch.Tensor:
        """Globally pooled activations from a ResNet stage before its FC head."""
        self.feature_dim(layer)
        backbone = self.backbone
        features = backbone.conv1(images)
        features = backbone.bn1(features)
        features = backbone.relu(features)
        features = backbone.maxpool(features)
        for name in self.FEATURE_DIMS:
            features = getattr(backbone, name)(features)
            if name == layer:
                return backbone.avgpool(features).flatten(1)
        raise AssertionError(f"unreachable layer {layer!r}")

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.backbone(images)
