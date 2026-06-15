import math
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

from app.search.base import SearchEngine
from app.text_processor import process_text


class TFIDFEngine(SearchEngine):
    def __init__(self) -> None:
        self.documents: List[Dict[str, Any]] = []
        self.tokenized_docs: List[List[str]] = []
        self.idf: Dict[str, float] = {}
        self.doc_vectors: List[Dict[str, float]] = []
        self.avg_dl = 0.0

    def index(self, docs: List[Dict[str, Any]]) -> None:
        self.documents = docs
        self.tokenized_docs = []
        for doc in docs:
            tokens = doc.get("tokens") or []
            if not tokens:
                proc = process_text(doc.get("content_raw", ""), doc.get("title", ""))
                tokens = proc["tokens"]
            self.tokenized_docs.append(tokens)

        self._build_idf()
        self._build_vectors()
        self.avg_dl = sum(len(t) for t in self.tokenized_docs) / max(len(self.tokenized_docs), 1)

    def _build_idf(self) -> None:
        doc_freq = defaultdict(int)
        total = len(self.tokenized_docs)
        for tokens in self.tokenized_docs:
            seen = set(tokens)
            for token in seen:
                doc_freq[token] += 1
        self.idf = {token: math.log((total + 1) / (freq + 1)) + 1 for token, freq in doc_freq.items()}

    def _build_vectors(self) -> None:
        self.doc_vectors = []
        for tokens in self.tokenized_docs:
            counts = Counter(tokens)
            total = len(tokens)
            vec = {}
            for token, count in counts.items():
                if token in self.idf:
                    tf = count / total if total else 0
                    vec[token] = tf * self.idf[token]
            self.doc_vectors.append(vec)

    def _vectorize(self, tokens: List[str]) -> Dict[str, float]:
        counts = Counter(tokens)
        total = len(tokens)
        vec = {}
        for token, count in counts.items():
            if token in self.idf:
                tf = count / total if total else 0
                vec[token] = tf * self.idf[token]
        return vec

    def _cosine(self, a: Dict[str, float], b: Dict[str, float]) -> float:
        norm_a = math.sqrt(sum(v * v for v in a.values()))
        norm_b = math.sqrt(sum(v * v for v in b.values()))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        dot = 0.0
        for token, weight in a.items():
            if token in b:
                dot += weight * b[token]
        return dot / (norm_a * norm_b)

    def search(
        self, query: str, source_filter: Optional[str] = None, limit: int = 10, offset: int = 0
    ) -> List[Dict[str, Any]]:
        if not self.doc_vectors:
            return []
        query_tokens = process_text(query).get("tokens", [])
        query_vec = self._vectorize(query_tokens)
        scored: List[Tuple[int, float]] = []
        for idx, doc_vec in enumerate(self.doc_vectors):
            doc = self.documents[idx]
            if source_filter and doc.get("source") != source_filter:
                continue
            score = self._cosine(query_vec, doc_vec)
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
        if not self.doc_vectors:
            return {"query": query, "algo": "tfidf", "tokens": [], "documents": [], "ranked": []}
        proc = process_text(query)
        query_tokens = proc.get("tokens", [])
        query_vec = self._vectorize(query_tokens)
        documents = []
        ranked = []
        for idx, doc_vec in enumerate(self.doc_vectors):
            doc = self.documents[idx]
            if source_filter and doc.get("source") != source_filter:
                continue
            score = self._cosine(query_vec, doc_vec)
            tokens = doc.get("tokens") or []
            term_counts = Counter(tokens)
            doc_vector_display = {term: round(weight, 6) for term, weight in doc_vec.items() if weight > 0}
            documents.append({
                "id": doc.get("id"),
                "title": doc.get("title", ""),
                "url": doc.get("url"),
                "source": doc.get("source"),
                "tokens": tokens[:50],
                "term_counts": dict(term_counts.most_common(20)),
                "vector": doc_vector_display,
                "score": round(score, 6) if score > 0 else 0,
            })
            if score > 0:
                ranked.append({**doc, "score": round(score, 6)})
        ranked.sort(key=lambda x: x["score"], reverse=True)
        return {
            "query": query,
            "algo": "tfidf",
            "tokens": query_tokens,
            "stemmed": proc.get("stemmed", []),
            "query_vector": {term: round(weight, 6) for term, weight in query_vec.items()},
            "idf": {term: round(weight, 6) for term, weight in self.idf.items() if term in query_tokens},
            "documents": documents[:limit],
            "ranked": ranked[:limit],
        }

    def reset(self) -> None:
        self.documents = []
        self.tokenized_docs = []
        self.idf = {}
        self.doc_vectors = []
