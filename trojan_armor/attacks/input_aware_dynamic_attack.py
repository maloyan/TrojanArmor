# attacks/input_aware_dynamic_attack.py
import torch
import torch.nn as nn
import numpy as np
from .base_attack import Attack


class TriggerGenerator(nn.Module):
    def __init__(self, channels):
        super(TriggerGenerator, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, channels, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class InputAwareDynamicAttack(Attack):
    def __init__(self, poisoning_rate, target_label, mask, device, epochs=50, learning_rate=0.001):
        super().__init__()
        self.poisoning_rate = poisoning_rate
        self.target_label = target_label
        self.mask = mask
        self.device = device
        self.epochs = epochs
        self.learning_rate = learning_rate

    def apply(self, images, labels):
        # Train the trigger generator
        trigger_generator = TriggerGenerator(images.size(1)).to(self.device)
        self.train_trigger_generator(trigger_generator, images)

        poisoned_images = []
        poisoned_labels = []

        for image, label in zip(images, labels):
            if np.random.random() < self.poisoning_rate:
                poisoned_image = self.apply_trigger(image, trigger_generator)
                poisoned_images.append(poisoned_image)
                poisoned_labels.append(self.target_label)
            else:
                poisoned_images.append(image)
                poisoned_labels.append(label)

        poisoned_images = torch.stack(poisoned_images)
        poisoned_labels = torch.tensor(poisoned_labels)

        return poisoned_images, poisoned_labels

    def train_trigger_generator(self, trigger_generator, images):
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(trigger_generator.parameters(), lr=self.learning_rate)

        for epoch in range(self.epochs):
            for image in images:
                image = image.unsqueeze(0).to(self.device)
                optimizer.zero_grad()

                output = trigger_generator(image)
                loss = criterion(output, image)

                loss.backward()
                optimizer.step()

    def apply_trigger(self, image, trigger_generator):
        with torch.no_grad():
            trigger = trigger_generator(image.unsqueeze(0).to(self.device)).squeeze(0)

        mask = torch.tensor(self.mask).float().to(self.device)
        poisoned_image = image * (1 - mask) + trigger * mask

        return poisoned_image
