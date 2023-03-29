# attacks/wanet.py
import torch
import torch.nn.functional as F

from .base_attack import Attack


class WaNet(Attack):
    def __init__(self, target_label, s=0.5, k=4, grid_rescale=1, attack_prob=None, **kwarg):
        super().__init__(attack_prob)
        self.target_label = target_label
        self.s = s
        self.k = k
        self.grid_rescale = grid_rescale

    def apply(self, images, labels):
        assert 0 <= self.attack_prob <= 1, "Attack probability must be in the range [0, 1]"

        batch_size = images.size(0)
        num_attacked = int(batch_size * self.attack_prob)

        for i in range(num_attacked):
            if torch.rand(1).item() < 0.5:
                images[i] = self.apply_attack(images[i], mode='attack')
                labels[i] = self.target_label
            else:
                images[i] = self.apply_attack(images[i], mode='noise')
        return images, labels

    def apply_attack(self, image, mode):

        img_size = image.shape[-1]
        ins = torch.rand(1, 2, self.k, self.k) * 2 - 1
        ins = ins / torch.mean(torch.abs(ins))
        noise_grid = (
            F.interpolate(ins, size=img_size, mode="bicubic", align_corners=True)
            .permute(0, 2, 3, 1)
            .to(image.device)
        )
        array1d = torch.linspace(-1, 1, steps=img_size)
        x, y = torch.meshgrid(array1d, array1d, indexing='xy')
        identity_grid = torch.stack((y, x), 2)[None, ...].to(image.device)

        grid_temps = (identity_grid + self.s * noise_grid / img_size) * self.grid_rescale
        grid_temps = torch.clamp(grid_temps, -1, 1)

        if mode=='attack':
            img = F.grid_sample(image.unsqueeze(0), grid_temps, align_corners=True)[0]
            return img

        ins = torch.rand(1, img_size, image.shape[1], 2) * 2 - 1
        grid_temps2 = grid_temps + ins / img_size
        grid_temps2 = torch.clamp(grid_temps2, -1, 1)
        img = F.grid_sample(image.unsqueeze(0), grid_temps2, align_corners=True)[0]
        return img

