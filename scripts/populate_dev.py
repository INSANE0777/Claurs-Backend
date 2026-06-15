import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["INDEXER_ENABLED"] = "false"
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///./nexus_search.db"
os.environ["DATABASE_URL_SYNC"] = "sqlite:///./nexus_search.db"

import asyncio
from datetime import datetime

from app.database import init_db, async_session_maker
from app.indexer import load_docs_from_db, sync_documents_to_db
from app.search import index_all


SAMPLE_DOCS = [
    {
        "url": "https://en.wikipedia.org/wiki/Python_(programming_language)",
        "title": "Python (programming language)",
        "content_raw": "Python is a high-level programming language. It supports multiple paradigms including structured, object-oriented, and functional programming.",
        "content_processed": "python high-level programming language supports multiple paradigms structured object-oriented functional programming",
        "tokens": ["python", "high-level", "programming", "language", "supports", "multiple", "paradigms", "structured", "object-oriented", "functional"],
        "source": "wikipedia",
        "content_hash": "hash1",
        "indexed_at": datetime.utcnow(),
    },
    {
        "url": "https://github.com/python/cpython",
        "title": "python/cpython",
        "content_raw": "The Python programming language. This is the official repository of CPython, the default implementation of Python.",
        "content_processed": "python programming language official repository cpython default implementation python",
        "tokens": ["python", "programming", "language", "official", "repository", "cpython", "default", "implementation"],
        "source": "github",
        "content_hash": "hash2",
        "indexed_at": datetime.utcnow(),
    },
    {
        "url": "https://www.reddit.com/r/programming/comments/python",
        "title": "Why Python is great for scripting",
        "content_raw": "Python is great for scripting and machine learning. The community is huge and libraries are mature.",
        "content_processed": "python great scripting machine learning community huge libraries mature",
        "tokens": ["python", "great", "scripting", "machine", "learning", "community", "huge", "libraries", "mature"],
        "source": "reddit",
        "content_hash": "hash3",
        "indexed_at": datetime.utcnow(),
    },
    {
        "url": "https://en.wikipedia.org/wiki/Machine_learning",
        "title": "Machine learning",
        "content_raw": "Machine learning is a branch of artificial intelligence that focuses on building systems that learn from data.",
        "content_processed": "machine learning branch artificial intelligence focuses building systems learn data",
        "tokens": ["machine", "learning", "branch", "artificial", "intelligence", "focuses", "building", "systems", "learn", "data"],
        "source": "wikipedia",
        "content_hash": "hash4",
        "indexed_at": datetime.utcnow(),
    },
    {
        "url": "https://github.com/pytorch/pytorch",
        "title": "pytorch/pytorch",
        "content_raw": "Tensors and dynamic neural networks in Python with strong GPU acceleration.",
        "content_processed": "tensors dynamic neural networks python strong gpu acceleration",
        "tokens": ["tensors", "dynamic", "neural", "networks", "python", "strong", "gpu", "acceleration"],
        "source": "github",
        "content_hash": "hash5",
        "indexed_at": datetime.utcnow(),
    },
]


async def main():
    await init_db()
    async with async_session_maker() as session:
        inserted = await sync_documents_to_db(session, SAMPLE_DOCS)
        docs = await load_docs_from_db(session)
    index_all(docs)
    print(f"Populated dev DB: {inserted} inserted/updated, {len(docs)} total docs.")


if __name__ == "__main__":
    asyncio.run(main())
