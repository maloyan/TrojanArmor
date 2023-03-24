# attacks/wanet.py
import numpy as np
import cv2
import torch
from .base_attack import Attack

class WaNet(Attack):
    def __init__(self, poisoning_rate, target_label, grid_size, max_offset):
        super().__init__() # type: ignore
        self.poisoning_rate = poisoning_rate
        self.target_label = target_label
        self.grid_size = grid_size
        self.max_offset = max_offset

    def apply(self, images, labels):
        poisoned_images = []
        poisoned_labels = []

        for image, label in zip(images, labels):
            if np.random.random() < self.poisoning_rate:
                poisoned_image = self.apply_warping(image)
                poisoned_images.append(poisoned_image)
                poisoned_labels.append(self.target_label)
            else:
                poisoned_images.append(image)
                poisoned_labels.append(label)

        poisoned_images = torch.stack(poisoned_images)
        poisoned_labels = torch.tensor(poisoned_labels)

        return poisoned_images, poisoned_labels

    def apply_warping(self, image):
        warping_field = self.generate_warping_field(image)
        warped_image_np = cv2.remap(image.cpu().numpy(), warping_field, None, cv2.INTER_LINEAR)

        warped_image_tensor = torch.tensor(warped_image_np).float()
        warped_image_tensor = warped_image_tensor.permute(2, 0, 1)

        return warped_image_tensor

    def generate_warping_field(self, image):
        k = self.grid_size
        h, w, _ = image.shape

        # Generate control points
        control_points_x = np.linspace(0, w, k)
        control_points_y = np.linspace(0, h, k)
        control_points = np.array(np.meshgrid(control_points_x, control_points_y)).T.reshape(-1, 2)

        # Add random offsets to control points
        control_points += np.random.uniform(-self.max_offset, self.max_offset, control_points.shape)

        # Perform bicubic interpolation
        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
        warping_field_x = cv2.resize(
            cv2.remap(grid_x, control_points, None, cv2.INTER_CUBIC),
            (w, h),
            interpolation=cv2.INTER_CUBIC
        )
        warping_field_y = cv2.resize(
            cv2.remap(grid_y, control_points, None, cv2.INTER_CUBIC),
            (w, h),
            interpolation=cv2.INTER_CUBIC
        )

        # Clip warping field to keep sampling points within the image boundary
        warping_field_x = np.clip(warping_field_x, 0, w - 1)
        warping_field_y = np.clip(warping_field_y, 0, h - 1)

        warping_field = np.stack([warping_field_x, warping_field_y], axis=-1).astype(np.float32)

        return warping_field
