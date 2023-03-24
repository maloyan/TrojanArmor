# attacks/sig.py
import torch
import numpy as np
from .base_attack import Attack


class SIG(Attack):
    def __init__(self, poisoning_rate, target_label, delta, signal_type, frequency=None):
        super().__init__()
        self.poisoning_rate = poisoning_rate
        self.target_label = target_label
        self.delta = delta
        self.signal_type = signal_type
        self.frequency = frequency

    def apply(self, images, labels):
        poisoned_images = []
        poisoned_labels = []

        for image, label in zip(images, labels):
            if np.random.random() < self.poisoning_rate:
                poisoned_image = self.apply_trigger(image)
                poisoned_images.append(poisoned_image)
                poisoned_labels.append(self.target_label)
            else:
                poisoned_images.append(image)
                poisoned_labels.append(label)

        poisoned_images = torch.stack(poisoned_images)
        poisoned_labels = torch.tensor(poisoned_labels)

        return poisoned_images, poisoned_labels

    def apply_trigger(self, image):
        if self.signal_type == "ramp":
            trigger = self.generate_ramp_trigger(image.size(1), image.size(2), self.delta)
        elif self.signal_type == "sinusoidal":
            trigger = self.generate_sinusoidal_trigger(image.size(1), image.size(2), self.delta, self.frequency)
        else:
            raise ValueError(f"Unsupported signal type: {self.signal_type}")

        poisoned_image = image + trigger
        return torch.clamp(poisoned_image, 0, 1)

    def generate_ramp_trigger(self, rows, columns, delta):
        trigger = torch.arange(columns, dtype=torch.float32).unsqueeze(0).repeat(rows, 1)
        trigger = delta * trigger / columns
        return trigger.unsqueeze(0)

    def generate_sinusoidal_trigger(self, rows, columns, delta, frequency):
        x = torch.arange(columns, dtype=torch.float32)
        trigger = delta * torch.sin(2 * np.pi * x * frequency / columns).unsqueeze(0).repeat(rows, 1)
        return trigger.unsqueeze(0)
