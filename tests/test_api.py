import os
import sys

# Ensure backend is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["INDEXER_ENABLED"] = "false"
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///./test_nexus.db"
os.environ["DATABASE_URL_SYNC"] = "sqlite:///./test_nexus.db"

import asyncio
from datetime import datetime

from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import async_session_maker, init_db
from app.indexer import load_docs_from_db, sync_documents_to_db
from app.models import Document, SearchLog
from app.search import index_all
from app.main import app


def add_test_documents():
    async def _inner():
        await init_db()
        docs = [
            {
                "url": "https://en.wikipedia.org/wiki/Python_(programming_language)",
                "title": "Python (programming language)",
                "content_raw": "Python is a high-level programming language. It supports multiple paradigms.",
                "content_processed": "python high-level programming language supports multiple paradigms",
                "tokens": ["python", "high-level", "programming", "language", "supports", "multiple", "paradigms"],
                "source": "wikipedia",
                "content_hash": "hash1",
                "indexed_at": datetime.utcnow(),
            },
            {
                "url": "https://github.com/python/cpython",
                "title": "cpython",
                "content_raw": "The Python programming language. Repository of CPython, the default implementation.",
                "content_processed": "python programming language repository cpython default implementation",
                "tokens": ["python", "programming", "language", "repository", "cpython", "default", "implementation"],
                "source": "github",
                "content_hash": "hash2",
                "indexed_at": datetime.utcnow(),
            },
            {
                "url": "https://www.reddit.com/r/programming/comments/python",
                "title": "Why Python is great",
                "content_raw": "Python is great for scripting and machine learning.",
                "content_processed": "python great scripting machine learning",
                "tokens": ["python", "great", "scripting", "machine", "learning"],
                "source": "reddit",
                "content_hash": "hash3",
                "indexed_at": datetime.utcnow(),
            },
        ]
        async with async_session_maker() as session:
            await sync_documents_to_db(session, docs)
            loaded = await load_docs_from_db(session)
        index_all(loaded)

    asyncio.run(_inner())


def test_health():
    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        print("health ok", data)


def test_search():
    with TestClient(app) as client:
        r = client.get("/search?q=python&algo=bm25")
        assert r.status_code == 200
        data = r.json()
        assert data["query"] == "python"
        assert data["algo"] == "bm25"
        assert len(data["results"]) > 0
        print("search ok", data["total"], data["results"][0]["title"])


def test_search_with_source():
    with TestClient(app) as client:
        r = client.get("/search?q=python&algo=bm25&source=github")
        assert r.status_code == 200
        data = r.json()
        for res in data["results"]:
            assert res["source"] == "github"
        print("source filter ok")


def test_autocomplete():
    with TestClient(app) as client:
        r = client.get("/autocomplete?prefix=pyth")
        assert r.status_code == 200
        data = r.json()
        assert "suggestions" in data
        print("autocomplete ok", data["suggestions"])


def test_analytics():
    with TestClient(app) as client:
        r = client.get("/analytics")
        assert r.status_code == 200
        data = r.json()
        assert "total_documents" in data
        print("analytics ok", data["total_documents"])


if __name__ == "__main__":
    add_test_documents()
    test_health()
    test_search()
    test_search_with_source()
    test_autocomplete()
    test_analytics()
    print("\nAll tests passed.")
