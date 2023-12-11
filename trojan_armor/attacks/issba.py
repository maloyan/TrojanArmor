# attacks/issba.py
import torch
import numpy as np
from .base_attack import Attack


class ISSBA(Attack):
    def __init__(self, poisoning_rate, target_label, encoder_decoder, **kwarg):
        super().__init__()
        self.poisoning_rate = poisoning_rate
        self.target_label = target_label
        self.encoder_decoder = encoder_decoder

    def apply(self, images, labels):
        poisoned_images = []
        poisoned_labels = []

        for image, label in zip(images, labels):
            if np.random.random() < self.poisoning_rate:
                poisoned_image = self.apply_issba_trigger(image)
                poisoned_images.append(poisoned_image)
                poisoned_labels.append(self.target_label)
            else:
                poisoned_images.append(image)
                poisoned_labels.append(label)

        poisoned_images = torch.stack(poisoned_images)
        poisoned_labels = torch.tensor(poisoned_labels)

        return poisoned_images, poisoned_labels

    def apply_issba_trigger(self, image):
        encoded_image = self.encoder_decoder.encode(image)
        trigger = self.encoder_decoder.decode(encoded_image)
        poisoned_image = image + trigger
        return torch.clamp(poisoned_image, 0, 1)
