from abc import ABC, abstractmethod
from typing import Tuple

class Reward(ABC):
    @abstractmethod
    def calculate_reward(self, observation, info) -> Tuple[float, bool]: pass
