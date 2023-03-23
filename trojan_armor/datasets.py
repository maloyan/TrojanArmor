# datasets.py

import torchvision
import torchvision.transforms as transforms

class DatasetHandler:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name

    def get_dataset(self, train=True, transform=None):
        if self.dataset_name == "mnist":
            dataset = torchvision.datasets.MNIST(
                root="./data",
                train=train,
                transform=transform,
                download=True
            )
        elif self.dataset_name == "cifar10":
            dataset = torchvision.datasets.CIFAR10(
                root="./data",
                train=train,
                transform=transform,
                download=True
            )
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")

        return dataset

    def get_transform(self):
        if self.dataset_name == "mnist":
            transform = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        elif self.dataset_name == "cifar10":
            transform = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")

        return transform

