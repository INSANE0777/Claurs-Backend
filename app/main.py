from contextlib import asynccontextmanager
from typing import AsyncGenerator

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from app.config import get_settings
from app.database import init_db, async_session_maker
from app.indexer import refresh_indices, run_indexer
from app.search import index_all
from app.api import search, autocomplete, crawl, analytics, health

settings = get_settings()

limiter = Limiter(key_func=get_remote_address, default_limits=[settings.RATE_LIMIT])

scheduler: AsyncIOScheduler | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    global scheduler

    await init_db()

    # Load existing documents into indices
    async with async_session_maker() as session:
        from app.indexer import load_docs_from_db
        docs = await load_docs_from_db(session)
        if docs:
            index_all(docs)

    # Recreate scheduler cleanly on each startup (handles tests and hot reloads)
    if scheduler is not None and scheduler.running:
        try:
            scheduler.shutdown(wait=False)
        except Exception:
            pass
    scheduler = AsyncIOScheduler()
    scheduler.start()

    if settings.INDEXER_ENABLED:
        scheduler.add_job(
            _scheduled_index_job,
            "interval",
            hours=settings.INDEXER_INTERVAL_HOURS,
            id="nexus_indexer",
            replace_existing=True,
        )
        # Run initial index if DB is empty
        async with async_session_maker() as session:
            from app.models import Document
            from sqlalchemy import select, func
            result = await session.execute(select(func.count(Document.id)))
            count = result.scalar() or 0
        if count == 0:
            try:
                await run_indexer()
            except Exception as e:
                print(f"Initial indexing failed: {e}")

    yield

    if scheduler is not None and scheduler.running:
        scheduler.shutdown()


async def _scheduled_index_job() -> None:
    try:
        await run_indexer()
    except Exception as e:
        print(f"Scheduled indexing failed: {e}")


app = FastAPI(
    title=settings.APP_NAME,
    description="A configurable full-stack search engine.",
    version="1.0.0",
    lifespan=lifespan,
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(search.router)
app.include_router(autocomplete.router)
app.include_router(crawl.router)
app.include_router(analytics.router)
app.include_router(health.router)


@app.get("/")
async def root():
    return {"app": settings.APP_NAME, "docs": "/docs"}
