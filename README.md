# Clarus Search — Backend

FastAPI backend for the Clarus Search engine. It crawls, indexes, and ranks documents from Wikipedia, Reddit, and GitHub using three selectable algorithms.

## Features

- **Async crawler** (`httpx` + `asyncio`) with configurable concurrency, robots.txt cache, and depth limit
- **Source adapters**: Wikipedia (MediaWiki), Reddit (search.json), GitHub (REST search), generic HTML
- **Text pipeline**: HTML strip, lowercase, punctuation removal, stopword removal, tokenization, Porter stemming
- **Search engines**: BM25, TF-IDF, and semantic (sentence-transformers + ChromaDB)
- **SQLite dev** database with Postgres-ready SQLAlchemy setup
- **FTS5** fallback triggers for full-text search
- **Background indexer** via APScheduler (24h interval)
- **Rate limiting** via slowapi (60 req/min per IP)
- **Caching**: Redis optional, in-memory LRU fallback

## Setup (with uv)

```bash
uv venv --python 3.11 .venv
uv pip install -r requirements.txt --python .venv/Scripts/python.exe
```

## Running locally

```bash
cp .env.example .env
.venv/Scripts/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

On first startup the app loads existing documents and, if the DB is empty, runs an initial crawl. To skip the crawl and seed sample documents:

```bash
.venv/Scripts/python scripts/populate_dev.py
export INDEXER_ENABLED=false
.venv/Scripts/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Tests

```bash
.venv/Scripts/python tests/test_api.py
```

## Docker

```bash
docker build -t clarus-backend .
docker run -p 8000:8000 clarus-backend
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check + engine doc counts |
| GET | `/search?q=...&algo=bm25&page=1&limit=10&source=...&live=1` | Paginated ranked results; `live=1` triggers a crawl fallback if results are sparse |
| GET | `/autocomplete?prefix=...` | Trie-based suggestions |
| POST | `/crawl` | Trigger manual crawl |
| GET | `/analytics` | Usage metrics |
| POST | `/analytics/click` | Track result click |
