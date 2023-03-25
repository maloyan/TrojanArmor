import torch
import unittest
import numpy as np
from trojan_armor.attacks import FilterAttack

class TestFilterAttack(unittest.TestCase):
    def test_filter_attack_nashville(self):
        images = torch.rand(5, 3, 32, 32)
        labels = torch.tensor([0, 1, 2, 3, 4])

        filter_attack = FilterAttack(1.0, "nashville")
        poisoned_images, poisoned_labels = filter_attack.apply(images, labels)

        self.assertTrue(torch.equal(labels, poisoned_labels))
        self.assertFalse(torch.equal(images, poisoned_images))

    def test_filter_attack_gotham(self):
        images = torch.rand(5, 3, 32, 32)
        labels = torch.tensor([0, 1, 2, 3, 4])

        filter_attack = FilterAttack(1.0, "gotham")
        poisoned_images, poisoned_labels = filter_attack.apply(images, labels)

        self.assertTrue(torch.equal(labels, poisoned_labels))
        self.assertFalse(torch.equal(images, poisoned_images))

    def test_filter_attack_no_attack(self):
        images = torch.rand(5, 3, 32, 32)
        labels = torch.tensor([0, 1, 2, 3, 4])

        filter_attack = FilterAttack(0.0, "nashville")
        poisoned_images, poisoned_labels = filter_attack.apply(images, labels)

        self.assertTrue(torch.equal(labels, poisoned_labels))
        self.assertTrue(torch.equal(images, poisoned_images))

    def test_filter_attack_invalid_filter(self):
        images = torch.rand(5, 3, 32, 32)
        labels = torch.tensor([0, 1, 2, 3, 4])

        filter_attack = FilterAttack(1.0, "invalid_filter")

        with self.assertRaises(ValueError) as context:
            poisoned_images, poisoned_labels = filter_attack.apply(images, labels)

        self.assertEqual(str(context.exception), "Unsupported filter: invalid_filter")

if __name__ == '__main__':
    unittest.main()
