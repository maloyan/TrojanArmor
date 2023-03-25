import unittest
import torch

from trojan_armor.attacks import BlendedAttack

class TestBlendedAttack(unittest.TestCase):
    def setUp(self):
        self.target_label = 5
        self.key_pattern = torch.ones((3, 32, 32))
        self.alpha_poisoning = 0.1
        self.blended_attack = BlendedAttack(target_label=self.target_label, alpha_poisoning=self.alpha_poisoning, key_pattern=self.key_pattern)

    def test_apply(self):
        images = torch.zeros((10, 3, 32, 32))
        labels = torch.randint(0, 10, (10,))
    
        expected_image = self.alpha_poisoning * self.key_pattern + (1 - self.alpha_poisoning) * images[0]
        attacked_images, attacked_labels = self.blended_attack.apply(images, labels)
        # Ensure that the first image (since attack_prob is 1.0) is blended with the key_pattern

        self.assertTrue(torch.allclose(attacked_images[0], expected_image), "The attacked image is not blended correctly")

        # Ensure that the attacked labels are changed to the target label
        for i in range(len(attacked_labels)):
            self.assertEqual(attacked_labels[i].item(), self.target_label, "Attacked label should be changed to target label")


        # Ensure that the remaining images are not changed
        for i in range(1, len(images)):
            self.assertTrue(torch.allclose(attacked_images[i], images[i]), f"Image {i} should not be changed")

    def test_blend_key_pattern(self):
        image = torch.rand((3, 32, 32))
        blended_image = self.blended_attack.blend_key_pattern(image, self.key_pattern, self.alpha_poisoning)
        expected_image = self.alpha_poisoning * self.key_pattern + (1 - self.alpha_poisoning) * image
        self.assertTrue(torch.allclose(blended_image, expected_image), "The blended image is not computed correctly")
