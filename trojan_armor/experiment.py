# experiment.py

import torch
from torch.utils.data import DataLoader
from trojan_armor.attacks import *
from trojan_armor.datasets import DatasetHandler
from trojan_armor.metrics import Metrics
from trojan_armor.models import get_model

def get_attack_classes():
    return {cls.__name__: cls for cls in Attack.__subclasses__()}

def evaluate(model, dataloader, device, attack=None):
    y_true, y_pred = [], []
    model.eval()
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            if attack:
                images, _ = attack.apply(images)
            
            outputs = model(images)
            _, predicted = outputs.max(1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    return y_true, y_pred

def run_experiment(dataset_name, model_name, attack_method, attack_params, device="cuda"):
    # Get dataset handler
    dataset_handler = DatasetHandler(dataset_name)

    # Get dataset
    transform = dataset_handler.get_transform()
    train_data = dataset_handler.get_dataset(train=True, transform=transform)
    test_data = dataset_handler.get_dataset(train=False, transform=transform)

    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

    # Get model
    model = get_model(model_name, num_classes=10).to(device)
    # Train model (add your training code here)

    y_true_before, y_pred_before = evaluate(model, test_loader, device)
    accuracy_before = Metrics.accuracy(y_true_before, y_pred_before)
    print(f"Accuracy before attack: {accuracy_before:.2f}")

    # Instantiate the attack object
    attack_classes = get_attack_classes()

    # Instantiate the attack object
    if attack_method in attack_classes:
        attack = attack_classes[attack_method](**attack_params)
    else:
        raise ValueError(f"Unsupported attack: {attack_method}")

    y_true_after, y_pred_after = evaluate(model, test_loader, device, attack)
    accuracy_after = Metrics.accuracy(y_true_after, y_pred_after)
    print(f"Accuracy after attack: {accuracy_after:.2f}")
    # Save results
