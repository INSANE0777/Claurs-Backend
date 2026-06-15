from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.autocomplete import get_suggestions
from app.database import get_db

router = APIRouter()


@router.get("/autocomplete")
async def autocomplete(
    prefix: str = Query(..., min_length=1, max_length=100),
    limit: int = Query(default=5, ge=1, le=20),
    db: AsyncSession = Depends(get_db),
):
    suggestions = await get_suggestions(db, prefix, limit)
    return {"prefix": prefix, "suggestions": suggestions}
