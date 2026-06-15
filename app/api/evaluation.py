from typing import List, Optional

from fastapi import APIRouter, Query

from app.evaluation import DEFAULT_ALGOS, DEFAULT_QUERIES, evaluate_all

router = APIRouter()


@router.get("/evaluation")
async def evaluation(
    q: Optional[List[str]] = Query(default=None, description="Optional list of test queries"),
    algo: List[str] = Query(default=DEFAULT_ALGOS, description="Algorithms to evaluate"),
    k: int = Query(default=10, ge=1, le=50, description="Evaluation cutoff"),
):
    queries = q if q else None
    result = evaluate_all(queries=queries, algos=algo, k=k)
    return result
