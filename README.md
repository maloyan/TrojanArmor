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
- [ ] Blended
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

| Method                                         | Trojan Armor | [Backdoor Toolbox](https://github.com/vtu81/backdoor-toolbox) | [BackdoorBench](https://github.com/SCLBD/BackdoorBench) | [BackdoorBox](https://github.com/THUYimingLi/BackdoorBox) | [TrojanZoo](https://github.com/ain-soph/trojanzoo) |
|------------------------------------------------|--------------|------------------|----------------|-------------|-----------|
| [BadNet (2017)](https://ieeexplore.ieee.org/document/8685687)                                  | ✅           | ✅               | ✅             | ✅          | ✅         |
| [Blended (2017)](https://arxiv.org/abs/1712.05526)                                             | ✅           | ✅               | ✅             | ✅          | ✅         |
| [TrojanNN (2017)](https://docs.lib.purdue.edu/cgi/viewcontent.cgi?article=2782&context=cstech) | ❌           | ✅               | ❌             | ❌          | ✅         |
| [Poison Frogs (2018)](https://proceedings.neurips.cc/paper/2018/hash/22722a343513ed45f14905eb07621686-Abstract.html)                                   | ❌           | ❌               | ❌             | ❌          | ❌         |
| [Filter Attack (2019)](https://dl.acm.org/doi/10.1145/3319535.3363216)                         | ✅           | ❌               | ❌             | ❌          | ❌         |
| [WaNet (2021)](https://arxiv.org/abs/2102.10369)                                               | ❌           | ✅               | ✅             | ✅          | ❌         |
| [Input Aware Dynamic Attack (2020)](https://arxiv.org/abs/2010.08138)                          | ❌           | ✅               | ✅             | ✅          | ✅         |
| [SIG (2019)](https://arxiv.org/abs/1902.11237)                                                 | ❌           | ✅               | ✅             | ❌          | ❌         |
| [Label Consistent Backdoor Attack (Clean Label) (2019)](https://arxiv.org/abs/1912.02771)      | ❌           | ✅               | ✅             | ✅          | ❌         |
| [ISSBA (2019)](https://arxiv.org/abs/1909.02742)                                               | ❌           | ✅               | ✅             | ✅          | ❌         |
| [IMC (2019)](https://arxiv.org/abs/1911.01559)                                                 | ❌           | ✅               | ❌             | ❌          | ✅         |
| [TrojanNet Attack (2020)](https://arxiv.org/abs/2002.10078)                                    | ❌           | ❌               | ❌             | ❌          | ✅         |
| [Refool (2020)](https://arxiv.org/abs/2007.02343)                                              | ❌           | ✅               | ❌             | ✅          | ✅         |
| [TaCT (2019)](https://arxiv.org/abs/1908.00686)                                                | ❌           | ✅               | ❌             | ❌          | ❌         |
| [Adaptive (2023)](https://openreview.net/forum?id=_wSHsgrVali)                                 | ❌           | ✅               | ❌             | ❌          | ❌         |
| [SleeperAgent (2022)](https://arxiv.org/abs/2106.08970)                                        | ❌           | ✅               | ❌             | ✅          | ❌         |
| [Low Frequency (2021)](https://openaccess.thecvf.com/content/ICCV2021/papers/Zeng_Rethinking_the_Backdoor_Attacks_Triggers_A_Frequency_Perspective_ICCV_2021_paper.pdf)                                  | ❌           | ❌               | ✅             | ❌          | ❌         |
| [TUAP (2020)](https://openaccess.thecvf.com/content_CVPR_2020/papers/Zhao_Clean-Label_Backdoor_Attacks_on_Video_Recognition_Models_CVPR_2020_paper.pdf)                                           | ❌           | ❌               | ❌             | ✅          | ❌         |
| [PhysicalBA (2021)](https://arxiv.org/abs/2104.02361)                                          | ❌           | ❌               | ❌             | ✅          | ❌         |
| [LIRA (2021)](https://openaccess.thecvf.com/content/ICCV2021/papers/Doan_LIRA_Learnable_Imperceptible_and_Robust_Backdoor_Attacks_ICCV_2021_paper.pdf)                                           | ❌           | ❌               | ❌             | ✅          | ❌         |
| [Blind (blended-based) (2020)](https://arxiv.org/abs/2005.03823)                               | ❌           | ❌               | ❌             | ✅          | ❌         |
| [LatentBackdoor (2019)](https://people.cs.uchicago.edu/~ravenben/publications/pdf/pbackdoor-ccs19.pdf)                                | ❌           | ❌               | ❌             | ❌          | ✅         |
| [Adversarial Embedding Attack (2019)](https://arxiv.org/abs/1905.13409)                        | ❌           | ❌               | ❌             | ❌          | ✅         |
