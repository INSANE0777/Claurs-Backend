from fastapi import APIRouter

from app.search import ENGINES

router = APIRouter()


@router.get("/health")
async def health():
    return {
        "status": "ok",
        "engines": {name: len(engine.documents) for name, engine in ENGINES.items()},
    }
