from typing import List, Optional

from sqlalchemy import select, update, delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import AutocompleteNode


class Trie:
    def __init__(self) -> None:
        self.root = {"children": {}, "is_word": False, "word": None, "weight": 0}

    def insert(self, word: str, weight: int = 1) -> None:
        node = self.root
        for ch in word.lower():
            if ch not in node["children"]:
                node["children"][ch] = {"children": {}, "is_word": False, "word": None, "weight": 0}
            node = node["children"][ch]
        node["is_word"] = True
        node["word"] = word.lower()
        node["weight"] += weight

    def _node_for(self, prefix: str) -> Optional[dict]:
        node = self.root
        for ch in prefix.lower():
            if ch not in node["children"]:
                return None
            node = node["children"][ch]
        return node

    def _collect(self, node: dict, results: List[dict], limit: int) -> None:
        if len(results) >= limit:
            return
        if node.get("is_word") and node.get("word"):
            results.append({"word": node["word"], "weight": node["weight"]})
        # Sort children by max weight descending for better suggestions
        for ch, child in sorted(node["children"].items(), key=lambda x: x[1].get("weight", 0), reverse=True):
            self._collect(child, results, limit)
            if len(results) >= limit:
                return

    def suggest(self, prefix: str, limit: int = 5) -> List[str]:
        node = self._node_for(prefix)
        if not node:
            return []
        results: List[dict] = []
        self._collect(node, results, limit)
        results.sort(key=lambda x: x["weight"], reverse=True)
        return [r["word"] for r in results[:limit]]


async def load_trie(session: AsyncSession) -> Trie:
    trie = Trie()
    result = await session.execute(select(AutocompleteNode))
    nodes = result.scalars().all()
    # Build in-memory trie from adjacency list (not strictly needed for suggest, but for persistence)
    for node in nodes:
        if node.is_word and node.word:
            trie.insert(node.word, node.weight)
    return trie


async def add_query(session: AsyncSession, query: str) -> None:
    query = query.strip().lower()
    if not query:
        return
    parent_id = None
    last_node = None
    for ch in query:
        stmt = select(AutocompleteNode).where(
            AutocompleteNode.parent_id == parent_id, AutocompleteNode.char == ch
        )
        result = await session.execute(stmt)
        node = result.scalar_one_or_none()
        if node is None:
            node = AutocompleteNode(parent_id=parent_id, char=ch, is_word=False, weight=0)
            session.add(node)
            await session.flush()
        parent_id = node.id
        last_node = node

    if last_node:
        last_node.is_word = True
        last_node.word = query
        last_node.weight += 1
        await session.commit()


async def get_suggestions(session: AsyncSession, prefix: str, limit: int = 5) -> List[str]:
    trie = await load_trie(session)
    return trie.suggest(prefix, limit)
