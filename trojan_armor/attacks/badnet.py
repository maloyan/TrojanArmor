# attacks/badnet.py
import torch
from .base_attack import Attack

class BadNet(Attack):
    def __init__(self, attack_params):
        super().__init__(attack_params)

    def apply(self, images):
        trigger = self.attack_params["trigger"]
        target_label = self.attack_params["target_label"]
        attack_prob = self.attack_params.get("attack_prob", 1.0)

        assert 0 <= attack_prob <= 1, "Attack probability must be in the range [0, 1]"

        batch_size = images.size(0)
        num_attacked = int(batch_size * attack_prob)

        for i in range(num_attacked):
            images[i] = self.apply_trigger(images[i], trigger)

        return images, torch.tensor([target_label] * num_attacked)

    def apply_trigger(self, image, trigger):
        trigger_h, trigger_w = trigger.size(1), trigger.size(2)
        image[:, -trigger_h:, -trigger_w:] = trigger
        return image
