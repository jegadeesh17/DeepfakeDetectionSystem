import torch
import torch.nn as nn
from torchvision import models

ARCHITECTURES = ("EfficientNet", "ResNet")


def build_efficientnet(num_classes: int = 1) -> nn.Module:
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def build_resnet(num_classes: int = 1) -> nn.Module:
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features, num_classes),
    )
    return model


def get_model(architecture: str, num_classes: int = 1) -> nn.Module:
    if architecture == "EfficientNet":
        return build_efficientnet(num_classes)
    if architecture == "ResNet":
        return build_resnet(num_classes)
    raise ValueError(f"Unknown architecture '{architecture}'. Expected one of {ARCHITECTURES}.")


def set_backbone_trainable(model: nn.Module, architecture: str, trainable: bool = False) -> None:
    """Freeze or unfreeze the pretrained backbone; classifier head always trains."""
    if architecture == "EfficientNet":
        for param in model.features.parameters():
            param.requires_grad = trainable
    elif architecture == "ResNet":
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True
        if trainable:
            for param in model.layer4.parameters():
                param.requires_grad = True
    else:
        raise ValueError(f"Unknown architecture '{architecture}'.")


def get_gradcam_target_layer(model: nn.Module, architecture: str):
    if architecture == "EfficientNet":
        return [model.features[-1]]
    if architecture == "ResNet":
        return [model.layer4[-1]]
    raise ValueError(f"Unknown architecture '{architecture}'.")


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
