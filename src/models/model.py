from abc import ABC, abstractmethod
from typing import Any


class Model(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def run(self):
        pass


class ModelFactory():
    def __init__(self) -> None:
        pass

    # Returns a list of models from a given config
    #
    # config must have this format:
    # [{"type": "model_name"}, ...]
    @classmethod
    def from_config(self, config: list[dict[str, Any]]) -> list["Model"]:
        return []

