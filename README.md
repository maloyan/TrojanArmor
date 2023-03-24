# TrojanArmor

<p align="center">
<img src="assets/logo.png" width=224 height=224>
</p>

TrojanArmor is a Python library for experimenting with trojan attacks in neural networks. With this library, you can choose the dataset, neural network architecture, and attack method for your experiments.

## Installation

Install the library using pip:

```bash
pip install git+https://github.com/maloyan/TrojanArmor.git
```

## Usage

To run an experiment, import the necessary modules and use the run_experiment function with the desired parameters:

```python
import torch
from trojan_armor.experiment import run_experiment

run_experiment(
    dataset_name="cifar10",
    model_name="timm_resnet18",
    attack_method="BadNet",
    attack_params={
        "trigger": torch.zeros(3, 5, 5),
        "target_label": 0,
        "attack_prob": 0.5,
    },
    device="cuda",
)
```

## Supported Datasets

- [ ] MNIST
- [ ] CIFAR-10
- [ ] CIFAR-100
- [ ] ImageNet
- [ ] GTSRB
- [ ] VGGFace2

## Supported Attack Methods

- [ ] BadNet
- [ ] TrojanNN
- [ ] Poison Frogs
- [ ] Filter Attack
- [ ] WaNet
- [ ] Input Aware Dynamic Attack
- [ ] SIG
- [ ] Label Consistent Backdoor Attack
- [ ] ISSBA
- [ ] IMC
- [ ] TrojanNet Attack
- [ ] Refool

## Supported Defense Methods

## Supported Neural Network Architectures

- All models from timm library

## Comparation with other libraries (WIP)

| Method                         | Trojan Armor | OpenBackdoor | Backdoor-toolbox | BackDoor Bench | BackdoorBox | TrojanZoo |
|--------------------------------|--------------|--------------|------------------|----------------|-------------|-----------|
| BadNet                         | ❌            | ✅            | ✅                | ✅              | ✅           | ✅         |
| TrojanNN                      | ❌            | ❌            | ✅                | ❌              | ❌           | ✅         |
| Poison Frogs                   | ❌            | ❌            | ❌                | ❌              | ❌           | ❌         |
| Filter Attack                  | ❌            | ❌            | ❌                | ❌              | ❌           | ❌         |
| WaNet                          | ❌            | ❌            | ✅                | ✅              | ❌           | ❌         |
| Input Aware Dynamic Attack     | ❌            | ❌            | ✅                | ✅              | ❌           | ❌         |
| SIG                            | ❌            | ❌            | ✅                | ✅              | ❌           | ❌         |
| Label Consistent Backdoor Attack | ❌          | ❌            | ❌                | ✅              | ✅           | ❌         |
| ISSBA                          | ❌            | ❌            | ✅                | ❌              | ❌           | ❌         |
| IMC                            | ❌            | ❌            | ❌                | ❌              | ❌           | ✅         |
| TrojanNet Attack               | ❌            | ❌            | ❌                | ❌              | ❌           | ❌         |
| Refool                         | ❌            | ❌            | ✅                | ❌              | ✅           | ❌         |
