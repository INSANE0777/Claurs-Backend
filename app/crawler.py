import asyncio
import hashlib
import urllib.robotparser
from datetime import datetime
from typing import Dict, List, Optional, Any, Set
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

from app.config import get_settings
from app.text_processor import process_text

settings = get_settings()

USER_AGENT = settings.CRAWLER_USER_AGENT
DEFAULT_HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate, br",
}


class RobotsCache:
    def __init__(self) -> None:
        self._cache: Dict[str, urllib.robotparser.RobotFileParser] = {}

    async def can_fetch(self, client: httpx.AsyncClient, url: str) -> bool:
        parsed = urlparse(url)
        robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
        if robots_url not in self._cache:
            parser = urllib.robotparser.RobotFileParser()
            parser.set_url(robots_url)
            try:
                response = await client.get(robots_url, timeout=10, follow_redirects=True)
                if response.status_code == 200:
                    parser.parse(response.text.splitlines())
                else:
                    parser.allow_all = True
            except Exception:
                parser.allow_all = True
            self._cache[robots_url] = parser
        return self._cache[robots_url].can_fetch(USER_AGENT, url)


class CrawlItem:
    __slots__ = ("url", "depth", "source")

    def __init__(self, url: str, depth: int, source: str) -> None:
        self.url = url
        self.depth = depth
        self.source = source


class BaseAdapter:
    name: str = "generic"

    async def fetch_documents(
        self, client: httpx.AsyncClient, item: CrawlItem
    ) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def extract_links(self, html: str, base_url: str) -> List[str]:
        soup = BeautifulSoup(html, "html.parser")
        links = []
        for a in soup.find_all("a", href=True):
            href = a["href"]
            full = urljoin(base_url, href)
            parsed = urlparse(full)
            if parsed.scheme in ("http", "https"):
                links.append(full)
        return links


class GenericAdapter(BaseAdapter):
    name = "generic"

    async def fetch_documents(
        self, client: httpx.AsyncClient, item: CrawlItem
    ) -> List[Dict[str, Any]]:
        response = await client.get(item.url, follow_redirects=True, timeout=settings.CRAWLER_REQUEST_TIMEOUT)
        response.raise_for_status()
        html = response.text
        title = ""
        soup = BeautifulSoup(html, "html.parser")
        if soup.title and soup.title.string:
            title = soup.title.string.strip()
        doc = {
            "url": item.url,
            "title": title,
            "content_raw": html,
            "source": item.source,
            "links": self.extract_links(html, item.url),
        }
        return [doc]


class WikipediaAdapter(BaseAdapter):
    name = "wikipedia"

    async def fetch_documents(
        self, client: httpx.AsyncClient, item: CrawlItem
    ) -> List[Dict[str, Any]]:
        # If seed is a search query, use MediaWiki search API.
        query = item.url.strip().replace(" ", "_")
        if not query.startswith("http"):
            search_url = "https://en.wikipedia.org/w/api.php"
            params = {
                "action": "query",
                "list": "search",
                "srsearch": query,
                "format": "json",
                "srlimit": 10,
            }
            response = await client.get(search_url, params=params, timeout=settings.CRAWLER_REQUEST_TIMEOUT)
            response.raise_for_status()
            data = response.json()
            results = data.get("query", {}).get("search", [])
            docs: List[Dict[str, Any]] = []
            for r in results:
                title = r.get("title", "")
                page_id = r.get("pageid")
                snippet = r.get("snippet", "")
                url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
                docs.append(
                    {
                        "url": url,
                        "title": title,
                        "content_raw": f"{title}\n{snippet}",
                        "source": "wikipedia",
                        "links": [url],
                        "meta": {"page_id": page_id},
                    }
                )
            return docs

        # If seed is a URL, fetch the page.
        response = await client.get(item.url, timeout=settings.CRAWLER_REQUEST_TIMEOUT)
        response.raise_for_status()
        html = response.text
        soup = BeautifulSoup(html, "html.parser")
        title = soup.title.string.strip() if soup.title and soup.title.string else ""
        body = soup.get_text(separator=" ", strip=True)
        return [
            {
                "url": item.url,
                "title": title,
                "content_raw": body,
                "source": "wikipedia",
                "links": self.extract_links(html, item.url),
            }
        ]


class RedditAdapter(BaseAdapter):
    name = "reddit"

    async def fetch_documents(
        self, client: httpx.AsyncClient, item: CrawlItem
    ) -> List[Dict[str, Any]]:
        query = item.url
        if not query.startswith("http"):
            query_url = f"https://www.reddit.com/search.json"
            params = {"q": query, "limit": 10, "sort": "relevance", "t": "all"}
        else:
            query_url = item.url
            params = {}

        headers = {**DEFAULT_HEADERS, "User-Agent": "Mozilla/5.0 NexusSearchBot/1.0"}
        response = await client.get(query_url, params=params, headers=headers, timeout=settings.CRAWLER_REQUEST_TIMEOUT)
        if response.status_code != 200:
            return []
        data = response.json()
        docs: List[Dict[str, Any]] = []
        posts = data.get("data", {}).get("children", [])
        for post in posts:
            p = post.get("data", {})
            title = p.get("title", "")
            selftext = p.get("selftext", "")
            url = p.get("permalink", "")
            if url:
                url = f"https://www.reddit.com{url}"
            docs.append(
                {
                    "url": url or p.get("url", ""),
                    "title": title,
                    "content_raw": f"{title}\n{selftext}",
                    "source": "reddit",
                    "links": [url] if url else [],
                    "meta": {"subreddit": p.get("subreddit"), "author": p.get("author")},
                }
            )
        return docs


class GitHubAdapter(BaseAdapter):
    name = "github"

    async def fetch_documents(
        self, client: httpx.AsyncClient, item: CrawlItem
    ) -> List[Dict[str, Any]]:
        query = item.url
        if not query.startswith("http"):
            # Search repositories
            search_url = "https://api.github.com/search/repositories"
            params = {"q": query, "per_page": 10, "sort": "stars", "order": "desc"}
        else:
            search_url = item.url
            params = {}

        headers = DEFAULT_HEADERS.copy()
        if settings.GITHUB_TOKEN:
            headers["Authorization"] = f"token {settings.GITHUB_TOKEN}"

        response = await client.get(search_url, params=params, headers=headers, timeout=settings.CRAWLER_REQUEST_TIMEOUT)
        if response.status_code != 200:
            return []
        data = response.json()
        docs: List[Dict[str, Any]] = []
        for repo in data.get("items", []):
            title = repo.get("full_name", "")
            description = repo.get("description") or ""
            url = repo.get("html_url", "")
            docs.append(
                {
                    "url": url,
                    "title": title,
                    "content_raw": f"{title}\n{description}\n{repo.get('language', '')}",
                    "source": "github",
                    "links": [url],
                    "meta": {
                        "stars": repo.get("stargazers_count"),
                        "language": repo.get("language"),
                    },
                }
            )
        return docs


class Crawler:
    def __init__(self, concurrency: int = settings.CRAWLER_CONCURRENCY) -> None:
        self.concurrency = concurrency
        self.max_depth = settings.CRAWLER_MAX_DEPTH
        self.delay = settings.CRAWLER_DELAY_SECONDS
        self.robots = RobotsCache()
        self.adapters: Dict[str, BaseAdapter] = {
            "generic": GenericAdapter(),
            "wikipedia": WikipediaAdapter(),
            "reddit": RedditAdapter(),
            "github": GitHubAdapter(),
        }
        self.seen: Set[str] = set()

    def _adapter_for(self, source: str) -> BaseAdapter:
        return self.adapters.get(source, self.adapters["generic"])

    def _hash(self, content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    async def _worker(
        self,
        queue: asyncio.Queue,
        client: httpx.AsyncClient,
        results: List[Dict[str, Any]],
        sem: asyncio.Semaphore,
    ) -> None:
        while True:
            try:
                item = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            if item.url in self.seen:
                queue.task_done()
                continue
            self.seen.add(item.url)
            async with sem:
                try:
                    if not await self.robots.can_fetch(client, item.url):
                        queue.task_done()
                        continue
                    await asyncio.sleep(self.delay)
                    adapter = self._adapter_for(item.source)
                    docs = await adapter.fetch_documents(client, item)
                    for doc in docs:
                        if not doc.get("url"):
                            continue
                        processed = process_text(doc.get("content_raw", ""), doc.get("title", ""))
                        doc["content_processed"] = processed["content_processed"]
                        doc["tokens"] = processed["tokens"]
                        doc["stemmed"] = processed["stemmed"]
                        doc["content_text"] = processed["content_text"]
                        doc["content_hash"] = self._hash(doc["content_raw"])
                        doc["indexed_at"] = datetime.utcnow()
                        results.append(doc)
                        # Enqueue discovered links for generic sources up to max depth
                        if item.depth < self.max_depth and item.source == "generic":
                            for link in doc.get("links", [])[:20]:
                                if link not in self.seen:
                                    await queue.put(CrawlItem(link, item.depth + 1, item.source))
                except Exception as e:
                    print(f"Crawl error for {item.url}: {e}")
                finally:
                    queue.task_done()

    async def crawl(
        self, seeds: List[Dict[str, Any]], max_depth: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        self.max_depth = max_depth if max_depth is not None else settings.CRAWLER_MAX_DEPTH
        queue: asyncio.Queue = asyncio.Queue()
        for seed in seeds:
            source = seed.get("source", "generic")
            url = seed.get("url") or seed.get("query") or ""
            await queue.put(CrawlItem(url, 0, source))

        results: List[Dict[str, Any]] = []
        sem = asyncio.Semaphore(self.concurrency)
        limits = httpx.Limits(max_connections=self.concurrency * 2, max_keepalive_connections=self.concurrency)
        async with httpx.AsyncClient(
            headers=DEFAULT_HEADERS, limits=limits, follow_redirects=True, timeout=settings.CRAWLER_REQUEST_TIMEOUT
        ) as client:
            workers = [
                asyncio.create_task(self._worker(queue, client, results, sem))
                for _ in range(self.concurrency)
            ]
            await queue.join()
            for w in workers:
                w.cancel()
            await asyncio.gather(*workers, return_exceptions=True)
        return results
