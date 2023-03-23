# experiment.py

import torch
from torch.utils.data import DataLoader
from models import get_model
from datasets import DatasetHandler
from attacks import Attack
from metrics import Metrics

def evaluate(model, dataloader, device):
    y_true, y_pred = [], []
    model.eval()
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

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
    model = get_model(model_name, num_classes=len(train_data.classes)).to(device)

    # Train model (add your training code here)

    y_true_before, y_pred_before = evaluate(model, test_loader, device)
    accuracy_before = Metrics.accuracy(y_true_before, y_pred_before)
    print(f"Accuracy before attack: {accuracy_before:.2f}")

    # Attack model
    attack = Attack(attack_method, attack_params)
    y_true, y_pred = [], []
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        images = attack.apply(images)

        # Evaluate attacked model
        with torch.no_grad():
            outputs = model(images)
            _, predicted = outputs.max(1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    accuracy = Metrics.accuracy(y_true, y_pred)
    print(f"Accuracy after attack: {accuracy_after:.2f}")
    # Save results

if __name__ == "__main__":
    run_experiment(
        dataset_name="cifar10",
        model_name="resnet18",
        attack_method="backdoor",
        attack_params={
            "patch": torch.zeros(3, 10, 10), "patch_size": 10, "position": (10, 10)},
        device="cuda"
    )

