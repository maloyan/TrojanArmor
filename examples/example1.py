import torch
from trojan_armor.experiment import run_experiment

def create_watermark(size=8):
    mark = torch.zeros(3, 32, 32)
    mark[:, -size:, -size:] = 1.0
    return mark

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

attack_filter = {
    "attack_prob": 0.5,
    "filter_name": "nashville"
}

attack_trojan_nn = {
    "mark": create_watermark(size=8),
    "preprocess_layer": "flatten",
    "preprocess_next_layer": "fc",
    "target_value": 100.0,
    "neuron_num": 2,
    "neuron_lr": 0.1,
    "neuron_epoch": 1000,
    "device": "cuda",
}

attack_wanet = {
    "target_label": 2,
    "attack_prob": 0.4
}

run_experiment(
    dataset_name="cifar10",
    model_name="timm_resnet18", #"hf_timm/resnetv2_50.a1h_in1k", #
    attack_method="WaNet",
    attack_params=attack_wanet,
    device="cuda",
    train_model=True
)
