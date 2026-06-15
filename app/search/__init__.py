from typing import Any, Dict, List, Optional

from app.search.base import SearchEngine
from app.search.tfidf import TFIDFEngine
from app.search.bm25 import BM25Engine
from app.search.semantic import SemanticEngine


ENGINES: Dict[str, SearchEngine] = {
    "tfidf": TFIDFEngine(),
    "bm25": BM25Engine(),
    "semantic": SemanticEngine(),
}


def get_engine(algo: str) -> SearchEngine:
    return ENGINES.get(algo.lower(), ENGINES["bm25"])


def trace(query: str, algo: str, source_filter: Optional[str] = None, limit: int = 10) -> Dict[str, Any]:
    engine = get_engine(algo)
    return engine.trace(query, source_filter=source_filter, limit=limit)


def index_all(docs: List[Dict[str, Any]]) -> None:
    for engine in ENGINES.values():
        engine.index(docs)


def reset_all() -> None:
    for engine in ENGINES.values():
        engine.reset()
