# attacks/base_attack.py
from abc import ABC, abstractmethod

class Attack(ABC):
    def __init__(self, attack_params):
        self.attack_params = attack_params

    @abstractmethod
    def apply(self, images, labels):
        pass