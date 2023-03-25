import torch
from trojan_armor.experiment import run_experiment

attack_params_badnet = {
    "trigger": torch.zeros(3, 5, 5),
    "target_label": 0,
    "attack_prob": 0.5,
}

attack_params_blended = {
    "target_label": 0,
    "key_pattern": torch.rand(3, 32, 32),
    "alpha_poisoning": 0.3,
    "attack_prob": 0.5,
}

run_experiment(
    dataset_name="cifar10",
    model_name="timm_resnet18", #"hf_timm/resnetv2_50.a1h_in1k", #
    attack_method="BlendedAttack",
    attack_params=attack_params_blended,
    device="cuda",
)