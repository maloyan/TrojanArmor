import torch
from trojan_armor.experiment import run_experiment

run_experiment(
    dataset_name="cifar10",
    model_name="resnet18",
    attack_method="BadNet",
    attack_params={
        "trigger": torch.zeros(3, 5, 5),
        "target_label": 0,
        "attack_prob": 0.5,
    },
    device="cuda",
)