import os

import torch
import torchvision
import unittest

from trojan_armor.defenses.transforms.randomized_smoothing import RandomizedSmoothing


TEST_DIR = os.path.dirname(__file__)


class TestRandomizedSmoothing(unittest.TestCase):
    def test_randomized_smoothing(self):
        img = torchvision.io.read_image(os.path.join(TEST_DIR, 'images/golden-retriever.jpg'))
        self.assertEqual(img.dtype, torch.uint8)

        images = img.resize(1, *img.shape)

        transform = RandomizedSmoothing()
        smoothed_images = transform(images)
        self.assertEqual(smoothed_images.dtype, torch.uint8)
