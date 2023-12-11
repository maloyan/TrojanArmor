# attacks/filter_attack.py
import torch
from .base_attack import Attack
import cv2
import numpy as np

class FilterAttack(Attack):
    def __init__(self, attack_prob, filter_name, **kwarg):
        self.attack_prob = attack_prob
        self.filter_name = filter_name

    def apply(self, images, labels):
        poisoned_images = []
        poisoned_labels = []
        for image, label in zip(images, labels):
            if np.random.random() < self.attack_prob:
                poisoned_image = self.apply_filter(image)
                poisoned_images.append(poisoned_image)
            else:
                poisoned_images.append(image)
            poisoned_labels.append(label)
        poisoned_images = torch.stack(poisoned_images)
        poisoned_labels = torch.tensor(poisoned_labels)

        return poisoned_images, poisoned_labels

    def apply_filter(self, image):
        if self.filter_name == "nashville":
            return self.apply_nashville_filter(image)
        elif self.filter_name == "gotham":
            return self.apply_gotham_filter(image)
        else:
            raise ValueError(f"Unsupported filter: {self.filter_name}")

    def apply_nashville_filter(self, image):
        image_np = image.cpu().numpy().transpose(1, 2, 0) * 255
        image_np = image_np.astype(np.uint8)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        # Implement the Nashville filter
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2HSV)
        image_np[:, :, 1] = cv2.add(image_np[:, :, 1], 45)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_HSV2BGR)

        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        image_tensor = torch.tensor(image_np).float() / 255
        image_tensor = image_tensor.permute(2, 0, 1)

        return image_tensor

    def apply_gotham_filter(self, image):
        image_np = image.cpu().numpy().transpose(1, 2, 0) * 255
        image_np = image_np.astype(np.uint8)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        # Implement the Gotham filter
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2HSV)
        image_np[:, :, 1] = cv2.add(image_np[:, :, 1], 50)
        image_np[:, :, 2] = cv2.add(image_np[:, :, 2], 50)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_HSV2BGR)

        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        image_tensor = torch.tensor(image_np).float() / 255
        image_tensor = image_tensor.permute(2, 0, 1)

        return image_tensor