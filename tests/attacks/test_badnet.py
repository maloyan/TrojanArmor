import torch
import unittest

from trojan_armor.attacks import BadNet

class TestBadNetAttack(unittest.TestCase):
    def test_apply(self):
        # Define test inputs
        batch_size = 5
        images = torch.randn(batch_size, 3, 32, 32)
        trigger = torch.ones(3, 8, 8)
        target_label = 1
        attack_prob = 0.6

        # Initialize BadNet attack
        badnet = BadNet(trigger, target_label, attack_prob)

        # Apply the attack
        attacked_images, target_labels = badnet.apply(images)

        # Test the output dimensions
        self.assertEqual(attacked_images.size(), images.size())
        self.assertEqual(len(target_labels), int(batch_size * attack_prob))

        # Test that the trigger has been applied to the images
        num_attacked = int(batch_size * attack_prob)
        for i in range(num_attacked):
            image_with_trigger = images[i].clone()
            image_with_trigger[:, -trigger.size(1):, -trigger.size(2):] = trigger
            self.assertTrue(torch.allclose(attacked_images[i], image_with_trigger))

        # Test the target labels
        self.assertTrue(torch.all(target_labels == target_label))

# if __name__ == '__main__':
#     unittest.main(argv=['first-arg-is-ignored'], exit=False)
