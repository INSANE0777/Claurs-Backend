from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from sqlalchemy import func, select, desc
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Document, SearchLog


async def get_top_queries(session: AsyncSession, limit: int = 10) -> List[Dict[str, Any]]:
    stmt = (
        select(SearchLog.query, func.count(SearchLog.id).label("count"))
        .group_by(SearchLog.query)
        .order_by(desc("count"))
        .limit(limit)
    )
    result = await session.execute(stmt)
    return [{"query": q, "count": c} for q, c in result.all()]


async def get_average_response_time(session: AsyncSession, hours: int = 24) -> float:
    since = datetime.utcnow() - timedelta(hours=hours)
    stmt = select(func.avg(SearchLog.response_time_ms)).where(SearchLog.timestamp >= since)
    result = await session.execute(stmt)
    val = result.scalar()
    return round(val, 2) if val else 0.0


async def get_total_documents(session: AsyncSession) -> int:
    result = await session.execute(select(func.count(Document.id)))
    return result.scalar() or 0


async def get_ctr_by_position(session: AsyncSession, max_position: int = 10) -> List[Dict[str, Any]]:
    """CTR per position = clicks at position / total searches."""
    total_searches = await session.execute(select(func.count(SearchLog.id)))
    total = total_searches.scalar() or 1
    stmt = (
        select(SearchLog.result_position, func.count(SearchLog.id).label("clicks"))
        .where(SearchLog.clicked_result_url.isnot(None))
        .group_by(SearchLog.result_position)
        .order_by(SearchLog.result_position)
    )
    result = await session.execute(stmt)
    rows = {pos: count for pos, count in result.all() if pos is not None}
    return [
        {"position": pos, "ctr": round(rows.get(pos, 0) / total, 4), "clicks": rows.get(pos, 0)}
        for pos in range(1, max_position + 1)
    ]


async def get_docs_over_time(session: AsyncSession, days: int = 7) -> List[Dict[str, Any]]:
    since = datetime.utcnow() - timedelta(days=days)
    stmt = (
        select(func.date(Document.indexed_at).label("day"), func.count(Document.id).label("count"))
        .where(Document.indexed_at >= since)
        .group_by("day")
        .order_by("day")
    )
    result = await session.execute(stmt)
    return [{"day": str(day), "count": count} for day, count in result.all()]


async def get_analytics(session: AsyncSession) -> Dict[str, Any]:
    return {
        "top_queries": await get_top_queries(session),
        "average_response_time_ms": await get_average_response_time(session),
        "total_documents": await get_total_documents(session),
        "ctr_by_position": await get_ctr_by_position(session),
        "docs_over_time": await get_docs_over_time(session),
    }
