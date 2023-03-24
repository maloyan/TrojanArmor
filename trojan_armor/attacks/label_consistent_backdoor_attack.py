# attacks/label_consistent_backdoor_attack.py
import torch
import numpy as np
from .base_attack import Attack

try:
    from art.attacks.evasion import FastGradientMethod
    art_available = True
except ImportError:
    art_available = False


class LabelConsistentBackdoorAttack(Attack):
    def __init__(self, poisoning_rate, target_label, g, alpha, epsilon, fgsm):
        super().__init__()
        self.poisoning_rate = poisoning_rate
        self.target_label = target_label
        self.g = g
        self.alpha = alpha
        self.epsilon = epsilon
        self.fgsm = fgsm

    def apply(self, images, labels):
        poisoned_images = []
        poisoned_labels = []

        for image, label in zip(images, labels):
            if np.random.random() < self.poisoning_rate:
                poisoned_image = self.apply_label_consistent_trigger(image)
                poisoned_images.append(poisoned_image)
                poisoned_labels.append(self.target_label)
            else:
                poisoned_images.append(image)
                poisoned_labels.append(label)

        poisoned_images = torch.stack(poisoned_images)
        poisoned_labels = torch.tensor(poisoned_labels)

        return poisoned_images, poisoned_labels

    def apply_label_consistent_trigger(self, image):
        z = torch.randn(1, self.g.nz, 1, 1, device=image.device)
        generated_image = self.g(z)

        if self.fgsm:
            if art_available:
                image = self.apply_art_fgsm(image)
            else:
                image = self.apply_fgsm(image)

        poisoned_image = self.alpha * generated_image + (1 - self.alpha) * image
        return torch.clamp(poisoned_image, 0, 1)

    def apply_fgsm(self, image):
        # Implement FGSM method here
        pass

    def apply_art_fgsm(self, image):
        # Implement ART FGSM method here
        pass
