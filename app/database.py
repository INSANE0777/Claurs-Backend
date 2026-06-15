from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base
from sqlalchemy.pool import NullPool

from app.config import get_settings

settings = get_settings()

Base = declarative_base()

engine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.DEBUG,
    poolclass=NullPool if "sqlite" in settings.DATABASE_URL else None,
)

async_session_maker = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
    autocommit=False,
)


async def get_db() -> AsyncSession:
    async with async_session_maker() as session:
        try:
            yield session
        finally:
            await session.close()


async def init_db() -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        # SQLite FTS5 fallback table + triggers for incremental updates
        if "sqlite" in settings.DATABASE_URL:
            await conn.exec_driver_sql(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS document_fts USING fts5(
                    title, content_raw, content_processed,
                    content='document', content_rowid='id',
                    tokenize='porter'
                )
                """
            )
            await conn.exec_driver_sql(
                """
                CREATE TRIGGER IF NOT EXISTS document_fts_insert AFTER INSERT ON document BEGIN
                    INSERT INTO document_fts(rowid, title, content_raw, content_processed)
                    VALUES (new.id, new.title, new.content_raw, new.content_processed);
                END
                """
            )
            await conn.exec_driver_sql(
                """
                CREATE TRIGGER IF NOT EXISTS document_fts_delete AFTER DELETE ON document BEGIN
                    INSERT INTO document_fts(document_fts, rowid, title, content_raw, content_processed)
                    VALUES ('delete', old.id, old.title, old.content_raw, old.content_processed);
                END
                """
            )
            await conn.exec_driver_sql(
                """
                CREATE TRIGGER IF NOT EXISTS document_fts_update AFTER UPDATE ON document BEGIN
                    INSERT INTO document_fts(document_fts, rowid, title, content_raw, content_processed)
                    VALUES ('delete', old.id, old.title, old.content_raw, old.content_processed);
                    INSERT INTO document_fts(rowid, title, content_raw, content_processed)
                    VALUES (new.id, new.title, new.content_raw, new.content_processed);
                END
                """
            )
