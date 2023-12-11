# attacks/blended.py
from .base_attack import Attack

class BlendedAttack(Attack):
    def __init__(self, target_label, alpha_poisoning, key_pattern, attack_prob=1.0, **kwarg):
        super().__init__(attack_prob=attack_prob)
        self.target_label = target_label
        self.alpha_poisoning = alpha_poisoning
        self.key_pattern = key_pattern
        self.attack_prob = attack_prob

    def apply(self, images, labels):
        assert 0 <= self.attack_prob <= 1, "Attack probability must be in the range [0, 1]"

        batch_size = images.size(0)
        num_attacked = int(batch_size * self.attack_prob)

        for i in range(num_attacked):
            images[i] = self.blend_key_pattern(images[i], key_pattern=self.key_pattern, alpha=self.alpha_poisoning)
            labels[i] = self.target_label
        return images, labels

    def blend_key_pattern(self, image, key_pattern, alpha):
        return alpha * key_pattern + (1 - alpha) * image
