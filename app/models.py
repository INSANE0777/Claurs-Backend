from datetime import datetime
from typing import Optional

from sqlalchemy import Column, Integer, String, Text, DateTime, Float, JSON, Boolean
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class Document(Base):
    __tablename__ = "document"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    url: Mapped[str] = mapped_column(String(2048), nullable=False, unique=True, index=True)
    title: Mapped[str] = mapped_column(String(1024), nullable=False, default="")
    content_raw: Mapped[str] = mapped_column(Text, nullable=False, default="")
    content_processed: Mapped[str] = mapped_column(Text, nullable=False, default="")
    tokens: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    source: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    indexed_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    meta: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)


class SearchLog(Base):
    __tablename__ = "search_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    query: Mapped[str] = mapped_column(String(512), nullable=False, index=True)
    algo: Mapped[str] = mapped_column(String(32), nullable=False, default="bm25")
    results_count: Mapped[int] = mapped_column(Integer, default=0)
    clicked_result_url: Mapped[Optional[str]] = mapped_column(String(2048), nullable=True)
    result_position: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    response_time_ms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    source_filter: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)


class AutocompleteNode(Base):
    __tablename__ = "autocomplete_node"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    parent_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True, index=True)
    char: Mapped[str] = mapped_column(String(1), nullable=False)
    is_word: Mapped[bool] = mapped_column(Boolean, default=False)
    weight: Mapped[int] = mapped_column(Integer, default=0, index=True)
    word: Mapped[Optional[str]] = mapped_column(String(512), nullable=True, index=True)

    __table_args__ = (
        {"sqlite_autoincrement": True},
    )
