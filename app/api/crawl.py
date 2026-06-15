from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Body
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.crawler import Crawler
from app.database import get_db
from app.indexer import sync_documents_to_db, DEFAULT_SEEDS

router = APIRouter()
settings = get_settings()


class CrawlRequest(BaseModel):
    url: Optional[str] = None
    source: Optional[str] = "generic"
    query: Optional[str] = None
    max_depth: Optional[int] = Field(default=None, ge=0, le=5)


class CrawlResponse(BaseModel):
    crawled_count: int
    inserted_or_updated: int


@router.post("/crawl", response_model=CrawlResponse)
async def crawl(
    payload: CrawlRequest = Body(...),
    db: AsyncSession = Depends(get_db),
):
    source = (payload.source or "generic").lower()
    if source not in {"wikipedia", "reddit", "github", "generic"}:
        raise HTTPException(status_code=400, detail="source must be one of: wikipedia, reddit, github, generic")

    url = payload.url or payload.query
    if not url:
        raise HTTPException(status_code=400, detail="Provide either 'url' or 'query'")

    seeds = [{"source": source, "url": url}]
    crawler = Crawler(concurrency=settings.CRAWLER_CONCURRENCY)
    crawled = await crawler.crawl(seeds, max_depth=payload.max_depth)
    inserted = await sync_documents_to_db(db, crawled)
    return CrawlResponse(crawled_count=len(crawled), inserted_or_updated=inserted)
