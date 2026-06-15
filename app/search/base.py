from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class SearchEngine(ABC):
    @abstractmethod
    def index(self, docs: List[Dict[str, Any]]) -> None:
        ...

    @abstractmethod
    def search(
        self, query: str, source_filter: Optional[str] = None, limit: int = 10, offset: int = 0
    ) -> List[Dict[str, Any]]:
        ...

    def reset(self) -> None:
        pass
