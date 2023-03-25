# attacks/base_attack.py
from abc import ABC, abstractmethod

class Attack(ABC):
    def __init__(self, attack_prob):
        self.attack_prob = attack_prob

    @abstractmethod
    def apply(self, images):
        pass