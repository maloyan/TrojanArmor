# models.py

import torch
import torchvision

def get_model(model_name, num_classes):
    if model_name == "resnet18":
        model = torchvision.models.resnet18(pretrained=True)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model

