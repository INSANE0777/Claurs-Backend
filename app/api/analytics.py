from fastapi import APIRouter, Depends, Body
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.analytics import get_analytics
from app.database import get_db
from app.models import SearchLog

router = APIRouter()


class ClickEvent(BaseModel):
    query: str
    url: str
    position: int


@router.get("/analytics")
async def analytics(db: AsyncSession = Depends(get_db)):
    return await get_analytics(db)


@router.post("/analytics/click")
async def click(
    payload: ClickEvent = Body(...),
    db: AsyncSession = Depends(get_db),
):
    log = SearchLog(
        query=payload.query[:512],
        algo="click",
        clicked_result_url=payload.url[:2048],
        result_position=payload.position,
    )
    db.add(log)
    await db.commit()
    return {"ok": True}
