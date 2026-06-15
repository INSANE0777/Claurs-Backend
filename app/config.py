import os
from functools import lru_cache
from typing import List, Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # App
    APP_NAME: str = "NexusSearch"
    DEBUG: bool = False

    # Database
    DATABASE_URL: str = "sqlite+aiosqlite:///./nexus_search.db"
    DATABASE_URL_SYNC: str = "sqlite:///./nexus_search.db"

    # Redis / Cache
    REDIS_URL: Optional[str] = None
    CACHE_TTL_SECONDS: int = 300
    CACHE_MAX_SIZE: int = 1000

    # Crawler
    CRAWLER_CONCURRENCY: int = 10
    CRAWLER_MAX_DEPTH: int = 2
    CRAWLER_REQUEST_TIMEOUT: int = 15
    CRAWLER_USER_AGENT: str = "NexusSearchBot/1.0 (+https://nexussearch.dev/bot)"
    CRAWLER_DELAY_SECONDS: float = 0.5

    # Indexer
    INDEXER_INTERVAL_HOURS: int = 24
    INDEXER_ENABLED: bool = True

    # Search
    SEARCH_DEFAULT_ALGO: str = "bm25"
    SEARCH_PAGE_SIZE: int = 10
    SEARCH_MAX_RESULTS: int = 100

    # Semantic
    SENTENCE_TRANSFORMER_MODEL: str = "all-MiniLM-L6-v2"
    CHROMA_PERSIST_DIR: str = "./chroma_db"

    # Rate limiting
    RATE_LIMIT: str = "60/minute"

    # Frontend
    FRONTEND_URL: str = "http://localhost:3000"
    ALLOWED_ORIGINS: List[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]

    # API keys / optional
    GITHUB_TOKEN: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = None

    # Sources
    SOURCES_ENABLED: List[str] = ["wikipedia", "reddit", "github"]

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
