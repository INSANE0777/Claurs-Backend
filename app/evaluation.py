import math
from typing import Any, Dict, List, Optional

DEFAULT_QUERIES = [
    "python",
    "backend",
    "machine learning",
    "github",
    "wikipedia",
    "deep learning",
    "neural network",
]

DEFAULT_ALGOS = ["bm25", "tfidf", "semantic"]


def _normalize(text: str) -> str:
    return text.lower().strip()


def auto_relevance(query: str, doc: Dict[str, Any]) -> int:
    """Binary relevance: 1 if all query tokens appear in the title, 0 otherwise."""
    query_tokens = _normalize(query).split()
    title = _normalize(doc.get("title", ""))
    if not query_tokens or not title:
        return 0
    if all(token in title for token in query_tokens):
        return 1
    return 0


def _dcg(relevances: List[float]) -> float:
    return sum((2 ** rel - 1) / math.log2(i + 2) for i, rel in enumerate(relevances))


def evaluate_query(
    query: str,
    algo: str,
    k: int = 10,
    relevance_fn=auto_relevance,
) -> Dict[str, Any]:
    from app.search import get_engine

    engine = get_engine(algo)
    results = engine.search(query, limit=k)
    relevances = [relevance_fn(query, r) for r in results]
    ideal = sorted(relevances, reverse=True)

    dcg_k = _dcg(relevances[:k])
    idcg_k = _dcg(ideal[:k])
    ndcg_k = dcg_k / idcg_k if idcg_k > 0 else 0.0

    mrr = 0.0
    for i, rel in enumerate(relevances):
        if rel > 0:
            mrr = 1.0 / (i + 1)
            break

    precision_k = sum(relevances[:k]) / k if k > 0 else 0.0

    all_docs = getattr(engine, "documents", []) or []
    total_relevant = sum(1 for d in all_docs if relevance_fn(query, d) > 0)
    recall_k = sum(relevances[:k]) / total_relevant if total_relevant > 0 else 0.0

    return {
        "query": query,
        "algo": algo,
        "ndcg": round(ndcg_k, 4),
        "mrr": round(mrr, 4),
        "precision": round(precision_k, 4),
        "recall": round(recall_k, 4),
        "num_results": len(results),
        "total_relevant": total_relevant,
    }


def evaluate_all(
    queries: Optional[List[str]] = None,
    algos: Optional[List[str]] = None,
    k: int = 10,
) -> Dict[str, Any]:
    queries = queries if queries else DEFAULT_QUERIES
    algos = algos if algos else DEFAULT_ALGOS

    per_query = []
    summary = {algo: {"ndcg": [], "mrr": [], "precision": [], "recall": []} for algo in algos}

    for query in queries:
        for algo in algos:
            result = evaluate_query(query, algo, k=k)
            per_query.append(result)
            for metric in ["ndcg", "mrr", "precision", "recall"]:
                summary[algo][metric].append(result[metric])

    averaged = {}
    for algo, metrics in summary.items():
        averaged[algo] = {
            metric: round(sum(values) / len(values), 4) if values else 0.0
            for metric, values in metrics.items()
        }

    return {
        "queries": queries,
        "algos": algos,
        "k": k,
        "averaged": averaged,
        "per_query": per_query,
    }
