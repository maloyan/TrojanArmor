# datasets/datasets.py
import torchvision
import torchvision.transforms as transforms
from .gtsrb import GTSRB
from .vggface2 import VGGFace2

class DatasetHandler:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name.lower()

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
        elif self.dataset_name == "cifar100":
            dataset = torchvision.datasets.CIFAR100(
                root="./data",
                train=train,
                transform=transform,
                download=True
            )
        elif self.dataset_name == "imagenet":
            dataset = torchvision.datasets.ImageNet(
                root="./data",
                split='train' if train else 'val',
                transform=transform,
                download=True
            )
        elif self.dataset_name == "gtsrb":
            dataset = GTSRB(
                root="./data",
                train=train,
                transform=transform,
                download=True
            )
        elif self.dataset_name == "vggface2":
            dataset = VGGFace2(
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
        elif self.dataset_name == "cifar100":
            transform = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        elif self.dataset_name == "imagenet":
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        elif self.dataset_name == "gtsrb":
            transform = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        elif self.dataset_name == "vggface2":
            transform = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")

        return
