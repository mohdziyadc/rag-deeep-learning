from abc import ABC, abstractmethod
from typing import Any
from models.parsed import ParsedDocument

class BaseParser(ABC):

    @abstractmethod
    def parse(
        self,
        name: str,
        data: bytes,
        metadata: dict[str, Any]
    ) -> ParsedDocument:
        raise NotImplementedError
