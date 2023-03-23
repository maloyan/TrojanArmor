# attacks.py

class Attack:
    def __init__(self, attack_method, attack_params):
        self.attack_method = attack_method
        self.attack_params = attack_params

    def apply(self, images):
        if self.attack_method == "backdoor":
            return self.backdoor(images, **self.attack_params)
        elif self.attack_method == "normal":
            return images
        # Add other attack methods here following the same structure
        else:
            raise ValueError(f"Unknown attack method: {self.attack_method}")

    def backdoor(self, images, patch, patch_size, position):
        x, y = position
        for i in range(images.size(0)):
            images[i, :, x:x+patch_size, y:y+patch_size] = patch
        return images

    # Add other attack methods here following the same structure

# Examples of how to add other attack methods:
#
#     @staticmethod
#     def BadNet(image, attack_params):
#         # Implement the BadNet attack
#         pass
#
#     @staticmethod
#     def TrojanNN(image, attack_params):
#         # Implement the TrojanNN attack
#         pass

