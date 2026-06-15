import math
from collections import Counter
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

    def trace(
        self, query: str, source_filter: Optional[str] = None, limit: int = 10
    ) -> Dict[str, Any]:
        if not self.bm25:
            return {"query": query, "algo": "bm25", "tokens": [], "documents": [], "ranked": []}
        proc = process_text(query)
        query_tokens = proc.get("tokens", [])
        if not query_tokens:
            return {"query": query, "algo": "bm25", "tokens": [], "documents": [], "ranked": []}

        k1, b = 1.5, 0.75
        N = len(self.tokenized_docs)
        avgdl = sum(len(t) for t in self.tokenized_docs) / N if N else 0
        doc_freq = {}
        for token in query_tokens:
            doc_freq[token] = sum(1 for tokens in self.tokenized_docs if token in tokens)
        idf = {token: math.log((N - doc_freq[token] + 0.5) / (doc_freq[token] + 0.5) + 1) for token in query_tokens}

        documents = []
        ranked = []
        for idx, tokens in enumerate(self.tokenized_docs):
            doc = self.documents[idx]
            if source_filter and doc.get("source") != source_filter:
                continue
            dl = len(tokens)
            term_counts = Counter(tokens)
            term_scores = {}
            total = 0.0
            for token in query_tokens:
                tf = term_counts.get(token, 0)
                denom = tf + k1 * (1 - b + b * (dl / avgdl)) if avgdl else 1
                term_score = idf[token] * (tf * (k1 + 1)) / denom if denom else 0
                term_scores[token] = round(term_score, 6)
                total += term_score
            doc_data = {
                "id": doc.get("id"),
                "title": doc.get("title", ""),
                "url": doc.get("url"),
                "source": doc.get("source"),
                "tokens": tokens[:50],
                "term_counts": dict(term_counts.most_common(20)),
                "doc_length": dl,
                "term_scores": term_scores,
                "score": round(total, 6) if total > 0 else 0,
            }
            documents.append(doc_data)
            if total > 0:
                ranked.append({**doc, "score": round(total, 6)})
        ranked.sort(key=lambda x: x["score"], reverse=True)
        return {
            "query": query,
            "algo": "bm25",
            "tokens": query_tokens,
            "stemmed": proc.get("stemmed", []),
            "avgdl": round(avgdl, 2),
            "idf": {token: round(val, 6) for token, val in idf.items()},
            "parameters": {"k1": k1, "b": b},
            "documents": documents[:limit],
            "ranked": ranked[:limit],
        }

    def reset(self) -> None:
        self.documents = []
        self.tokenized_docs = []
        self.bm25 = None
