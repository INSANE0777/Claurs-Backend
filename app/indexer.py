import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.crawler import Crawler
from app.database import async_session_maker
from app.models import Document
from app.search import index_all, reset_all
from app.cache import cache_manager

settings = get_settings()


DEFAULT_SEEDS = [
    {"source": "wikipedia", "url": "machine learning"},
    {"source": "wikipedia", "url": "artificial intelligence"},
    {"source": "wikipedia", "url": "python programming language"},
    {"source": "reddit", "url": "programming"},
    {"source": "reddit", "url": "machinelearning"},
    {"source": "github", "url": "fastapi"},
    {"source": "github", "url": "machine learning"},
]


async def load_docs_from_db(session: AsyncSession) -> List[Dict[str, Any]]:
    result = await session.execute(select(Document))
    docs = result.scalars().all()
    return [
        {
            "id": d.id,
            "url": d.url,
            "title": d.title,
            "content_raw": d.content_raw,
            "content_processed": d.content_processed,
            "tokens": d.tokens or [],
            "source": d.source,
            "indexed_at": d.indexed_at,
            "content_hash": d.content_hash,
            "meta": d.meta,
        }
        for d in docs
    ]


async def sync_documents_to_db(session: AsyncSession, crawled: List[Dict[str, Any]]) -> int:
    """Upsert crawled documents. Skip unchanged pages via content_hash."""
    result = await session.execute(select(Document.url, Document.content_hash))
    existing = {url: ch for url, ch in result.all()}
    inserted = 0
    for doc in crawled:
        url = doc["url"]
        content_hash = doc["content_hash"]
        if existing.get(url) == content_hash:
            continue
        if url in existing:
            await session.execute(delete(Document).where(Document.url == url))
        session.add(
            Document(
                url=url,
                title=doc.get("title", ""),
                content_raw=doc.get("content_raw", ""),
                content_processed=doc.get("content_processed", ""),
                tokens=doc.get("tokens", []),
                source=doc.get("source", "generic"),
                indexed_at=doc.get("indexed_at", datetime.utcnow()),
                content_hash=content_hash,
                meta=doc.get("meta"),
            )
        )
        inserted += 1
    await session.commit()
    return inserted


async def run_indexer(seeds: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    seeds = seeds or DEFAULT_SEEDS
    enabled = set(settings.SOURCES_ENABLED)
    seeds = [s for s in seeds if s.get("source") in enabled]

    crawler = Crawler(concurrency=settings.CRAWLER_CONCURRENCY)
    crawled = await crawler.crawl(seeds)

    async with async_session_maker() as session:
        inserted = await sync_documents_to_db(session, crawled)
        docs = await load_docs_from_db(session)

    index_all(docs)
    await cache_manager.invalidate_pattern("search:*")
    return {
        "crawled_count": len(crawled),
        "inserted_or_updated": inserted,
        "total_indexed": len(docs),
    }


async def refresh_indices() -> None:
    async with async_session_maker() as session:
        docs = await load_docs_from_db(session)
    index_all(docs)
