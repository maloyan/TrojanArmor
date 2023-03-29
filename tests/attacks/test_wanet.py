import unittest
import torch
from trojan_armor.attacks import WaNet

class TestWaNet(unittest.TestCase):

    def setUp(self):
        self.attack = WaNet(target_label=8, s=0.5, k=4, grid_rescale=1, attack_prob=0.1)

    def test_apply_attack(self):
        input_image = torch.randn(3, 32, 32)
        attacked_image = self.attack.apply_attack(input_image, mode='attack')
        self.assertEqual(attacked_image.shape, input_image.shape, "Output image shape should match input image shape")

    def test_apply(self):
        input_images = torch.randn(10, 3, 32, 32)
        input_labels = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        output_images, output_labels = self.attack.apply(input_images, input_labels)

        self.assertEqual(output_images.shape, input_images.shape, "Output images shape should match input images shape")
        self.assertEqual(output_labels.shape, input_labels.shape, "Output labels shape should match input labels shape")

        num_modified_labels = (output_labels != input_labels).sum().item()
        self.assertLessEqual(num_modified_labels, int(self.attack.attack_prob * len(input_labels)), "Number of modified labels should be less than or equal to attack_prob * batch_size")

