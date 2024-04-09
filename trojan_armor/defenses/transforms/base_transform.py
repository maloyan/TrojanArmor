from abc import ABC, abstractmethod

import torch


class BaseTransform(ABC):
    @abstractmethod
    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        return images
