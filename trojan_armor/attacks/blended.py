# attacks/blended.py
import numpy as np
from .base_attack import Attack

class BlendedAttack(Attack):
    def __init__(self, attack_params, key_pattern):
        super().__init__(attack_params=attack_params)
        self.key_pattern = key_pattern

    def apply(self, images, *, mode='poisoning'):
        if mode not in ['poisoning', 'backdoor']:
            raise ValueError("Invalid mode. Choose either 'poisoning' or 'backdoor'")
        blended_images = []
        for image in images:
            alpha = self.get_alpha(mode=mode)
            blended_image = self.blend_key_pattern(image, key_pattern=self.key_pattern, alpha=alpha)
            blended_images.append(blended_image)
        return np.array(blended_images)

    def get_alpha(self, *, mode):
        if mode == 'poisoning':
            return self.attack_params['alpha_poisoning']
        elif mode == 'backdoor':
            return self.attack_params['alpha_backdoor']
        else:
            raise ValueError("Invalid mode. Choose either 'poisoning' or 'backdoor'")

    def blend_key_pattern(self, image, *, key_pattern, alpha):
        return alpha * key_pattern + (1 - alpha) * image