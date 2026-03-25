from abc import ABC, abstractmethod


class Prompt(ABC):
    @abstractmethod
    def parse_assistant_content(self, assistant_content: str, **kwargs) -> dict: ...
