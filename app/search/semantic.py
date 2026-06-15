import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from app.config import get_settings
from app.search.base import SearchEngine
from app.text_processor import process_text

settings = get_settings()


class SemanticEngine(SearchEngine):
    def __init__(self) -> None:
        self.documents: List[Dict[str, Any]] = []
        self.vectors: Optional[np.ndarray] = None
        self.model = None
        self._model_name = settings.SENTENCE_TRANSFORMER_MODEL

    def _load_model(self):
        if self.model is None:
            from sentence_transformers import SentenceTransformer
            os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
            self.model = SentenceTransformer(self._model_name)
        return self.model

    def index(self, docs: List[Dict[str, Any]]) -> None:
        self.documents = docs
        if not docs:
            self.vectors = None
            return
        model = self._load_model()
        texts = []
        for doc in docs:
            title = doc.get("title", "")
            body = doc.get("content_processed") or doc.get("content_text", "")[:500]
            texts.append(f"{title}. {body}".strip() or doc.get("url", ""))
        self.vectors = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

    def search(
        self, query: str, source_filter: Optional[str] = None, limit: int = 10, offset: int = 0
    ) -> List[Dict[str, Any]]:
        if self.vectors is None or self.vectors.shape[0] == 0:
            return []
        model = self._load_model()
        query_vec = model.encode([query], convert_to_numpy=True, show_progress_bar=False)
        # cosine similarity
        norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
        q_norm = np.linalg.norm(query_vec, axis=1, keepdims=True)
        denom = (norms * q_norm.T) + 1e-10
        sims = np.dot(self.vectors, query_vec.T).flatten() / denom.flatten()
        scored: List[Tuple[int, float]] = []
        for idx, score in enumerate(sims):
            doc = self.documents[idx]
            if source_filter and doc.get("source") != source_filter:
                continue
            if score > 0:
                scored.append((idx, float(score)))
        scored.sort(key=lambda x: x[1], reverse=True)
        results = []
        for idx, score in scored[offset : offset + limit]:
            doc = self.documents[idx]
            results.append({**doc, "score": round(score, 6)})
        return results

    def trace(
        self, query: str, source_filter: Optional[str] = None, limit: int = 10
    ) -> Dict[str, Any]:
        if self.vectors is None or self.vectors.shape[0] == 0:
            return {"query": query, "algo": "semantic", "documents": [], "ranked": []}
        model = self._load_model()
        query_vec = model.encode([query], convert_to_numpy=True, show_progress_bar=False)
        norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
        q_norm = np.linalg.norm(query_vec, axis=1, keepdims=True)
        denom = (norms * q_norm.T) + 1e-10
        sims = np.dot(self.vectors, query_vec.T).flatten() / denom.flatten()
        documents = []
        ranked = []
        for idx, score in enumerate(sims):
            doc = self.documents[idx]
            if source_filter and doc.get("source") != source_filter:
                continue
            doc_embedding = self.vectors[idx].tolist()
            documents.append({
                "id": doc.get("id"),
                "title": doc.get("title", ""),
                "url": doc.get("url"),
                "source": doc.get("source"),
                "embedding_preview": [round(x, 4) for x in doc_embedding[:20]],
                "cosine_similarity": round(float(score), 6),
                "score": round(float(score), 6) if score > 0 else 0,
            })
            if score > 0:
                ranked.append({**doc, "score": round(float(score), 6)})
        ranked.sort(key=lambda x: x["score"], reverse=True)
        return {
            "query": query,
            "algo": "semantic",
            "model": self._model_name,
            "query_embedding_preview": [round(x, 4) for x in query_vec.flatten().tolist()[:20]],
            "documents": documents[:limit],
            "ranked": ranked[:limit],
        }

    def reset(self) -> None:
        self.documents = []
        self.vectors = None
