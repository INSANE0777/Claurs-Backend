import time
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from app.analytics import get_analytics
from app.autocomplete import add_query
from app.cache import cache_manager
from app.config import get_settings
from app.database import get_db
from app.indexer import run_live_crawl
from app.models import Document, SearchLog
from app.search import get_engine
from app.text_processor import build_snippet, process_text

router = APIRouter()
settings = get_settings()


ALGORITHMS = {"tfidf", "bm25", "semantic"}


LIVE_CRAWL_THRESHOLD = 3


@router.get("/search")
async def search(
    q: str = Query(..., min_length=1, description="Search query"),
    algo: str = Query(default=settings.SEARCH_DEFAULT_ALGO),
    page: int = Query(default=1, ge=1),
    limit: int = Query(default=settings.SEARCH_PAGE_SIZE, ge=1, le=100),
    source: Optional[str] = Query(default=None),
    live: bool = Query(default=False, description="Trigger live crawl if results are sparse"),
    request: Request = None,
    db: AsyncSession = Depends(get_db),
):
    algo = algo.lower()
    if algo not in ALGORITHMS:
        raise HTTPException(status_code=400, detail=f"Unknown algo '{algo}'. Use one of {ALGORITHMS}")

    start = time.perf_counter()
    offset = (page - 1) * limit
    cache_key = cache_manager.make_key("search", q, algo, source, page, limit, live)
    cached = await cache_manager.get(cache_key)

    results = []
    total = 0
    crawled = None

    if cached:
        results = cached["results"]
        total = cached["total"]
    else:
        engine = get_engine(algo)
        raw_results = engine.search(q, source_filter=source, limit=limit, offset=offset)
        query_tokens = process_text(q).get("tokens", [])
        total = len(engine.search(q, source_filter=source, limit=10000, offset=0)) if raw_results else 0
        results = []
        for r in raw_results:
            snippet = build_snippet(r.get("content_text", r.get("content_raw", "")), query_tokens)
            results.append({
                "id": r.get("id"),
                "title": r.get("title", "") or r.get("url", ""),
                "url": r.get("url"),
                "snippet": snippet,
                "source": r.get("source"),
                "score": r.get("score"),
                "indexed_at": r.get("indexed_at").isoformat() if r.get("indexed_at") else None,
            })

        # Live crawl fallback: if enabled and few/no results, crawl and re-search
        if live and total < LIVE_CRAWL_THRESHOLD:
            sources_to_crawl = [source] if source else settings.SOURCES_ENABLED
            try:
                crawled = await run_live_crawl(q, sources_to_crawl, max_depth=1)
                # Re-run search after indexing new docs
                raw_results = engine.search(q, source_filter=source, limit=limit, offset=offset)
                total = len(engine.search(q, source_filter=source, limit=10000, offset=0)) if raw_results else 0
                results = []
                for r in raw_results:
                    snippet = build_snippet(r.get("content_text", r.get("content_raw", "")), query_tokens)
                    results.append({
                        "id": r.get("id"),
                        "title": r.get("title", "") or r.get("url", ""),
                        "url": r.get("url"),
                        "snippet": snippet,
                        "source": r.get("source"),
                        "score": r.get("score"),
                        "indexed_at": r.get("indexed_at").isoformat() if r.get("indexed_at") else None,
                    })
            except Exception as e:
                crawled = {"error": str(e)}

        await cache_manager.set(cache_key, {"results": results, "total": total}, ttl=60 if live else None)

    response_time_ms = round((time.perf_counter() - start) * 1000, 2)

    # Log query
    log = SearchLog(
        query=q[:512],
        algo=algo,
        results_count=total,
        response_time_ms=response_time_ms,
        source_filter=source,
    )
    db.add(log)
    await add_query(db, q)
    await db.commit()

    # Did you mean suggestion: find closest query in top_queries by edit distance
    suggestion = await _suggest_spelling(db, q)

    return {
        "query": q,
        "algo": algo,
        "page": page,
        "limit": limit,
        "total": total,
        "results": results,
        "response_time_ms": response_time_ms,
        "suggestion": suggestion,
        "live": live,
        "crawl_info": crawled,
    }


@router.get("/search/trace")
async def search_trace(
    q: str = Query(..., min_length=1, description="Search query to trace"),
    algo: str = Query(default=settings.SEARCH_DEFAULT_ALGO),
    source: Optional[str] = Query(default=None),
    limit: int = Query(default=10, ge=1, le=50),
):
    algo = algo.lower()
    if algo not in ALGORITHMS:
        raise HTTPException(status_code=400, detail=f"Unknown algo '{algo}'. Use one of {ALGORITHMS}")
    from app.search import trace

    result = trace(q, algo=algo, source_filter=source, limit=limit)
    return result


async def _suggest_spelling(db: AsyncSession, q: str) -> Optional[str]:
    try:
        from difflib import get_close_matches
        stmt = select(SearchLog.query).distinct().limit(1000)
        result = await db.execute(stmt)
        queries = [r[0] for r in result.all() if r[0]]
        matches = get_close_matches(q.lower(), [q.lower() for q in queries], n=1, cutoff=0.75)
        if matches and matches[0] != q.lower():
            return matches[0]
    except Exception:
        pass
    return None
