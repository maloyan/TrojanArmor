# attacks/badnet.py
import torch
from .base_attack import Attack

class BadNet(Attack):
    def __init__(self, trigger, target_label, attack_prob=1.0):
        self.trigger = trigger
        self.target_label = target_label
        self.attack_prob = attack_prob

    def apply(self, images):
        assert 0 <= self.attack_prob <= 1, "Attack probability must be in the range [0, 1]"

        batch_size = images.size(0)
        num_attacked = int(batch_size * self.attack_prob)

        for i in range(num_attacked):
            images[i] = self.apply_trigger(images[i], self.trigger)

        return images, torch.tensor([self.target_label] * num_attacked)

    def apply_trigger(self, image, trigger):
        trigger_h, trigger_w = trigger.size(1), trigger.size(2)
        image[:, -trigger_h:, -trigger_w:] = trigger
        return image
