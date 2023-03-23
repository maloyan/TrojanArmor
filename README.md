# TrojanArmor

TrojanArmor is a Python library for experimenting with trojan attacks in neural networks. With this library, you can choose the dataset, neural network architecture, and attack method for your experiments.

## Installation

Install the library using pip:

```bash
pip install git+https://github.com/yourusername/TrojanArmor.git
```

## Usage

To run an experiment, import the necessary modules and use the run_experiment function with the desired parameters:

```
from trojan_armor.experiment import run_experiment

run_experiment(
    dataset_name="cifar10",
    model_name="resnet18",
    attack_method="backdoor",
    attack_params={
        "patch": torch.zeros(3, 10, 10), "patch_size": 10, "position": (10, 10)
    },
    device="cuda"
)
```

## Supported Datasets

- MNIST
- CIFAR-10

## Supported Neural Network Architectures

- ResNet18

## Supported Attack Methods

- [ ] BadNet
- [ ] TrojanNN
- [ ] Poison Frogs

## Supported Defense Methods
