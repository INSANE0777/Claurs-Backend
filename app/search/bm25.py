from typing import Any, Dict, List, Optional, Tuple

from rank_bm25 import BM25Okapi

from app.search.base import SearchEngine
from app.text_processor import process_text


class BM25Engine(SearchEngine):
    def __init__(self) -> None:
        self.documents: List[Dict[str, Any]] = []
        self.tokenized_docs: List[List[str]] = []
        self.bm25: Optional[BM25Okapi] = None

    def index(self, docs: List[Dict[str, Any]]) -> None:
        self.documents = docs
        self.tokenized_docs = []
        for doc in docs:
            tokens = doc.get("tokens") or []
            if not tokens:
                proc = process_text(doc.get("content_raw", ""), doc.get("title", ""))
                tokens = proc["tokens"]
            self.tokenized_docs.append(tokens)
        if self.tokenized_docs:
            self.bm25 = BM25Okapi(self.tokenized_docs)
        else:
            self.bm25 = None

    def search(
        self, query: str, source_filter: Optional[str] = None, limit: int = 10, offset: int = 0
    ) -> List[Dict[str, Any]]:
        if not self.bm25:
            return []
        query_tokens = process_text(query).get("tokens", [])
        if not query_tokens:
            return []
        scores = self.bm25.get_scores(query_tokens)
        scored: List[Tuple[int, float]] = []
        for idx, score in enumerate(scores):
            doc = self.documents[idx]
            if source_filter and doc.get("source") != source_filter:
                continue
            if score > 0:
                scored.append((idx, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        results = []
        for idx, score in scored[offset : offset + limit]:
            doc = self.documents[idx]
            results.append({**doc, "score": round(score, 6)})
        return results

    def reset(self) -> None:
        self.documents = []
        self.tokenized_docs = []
        self.bm25 = None
