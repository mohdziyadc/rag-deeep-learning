import os
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

# Need this imports for the Base.metadata.create_all to work
from core.models import MemoryModel, TaskModel

PSQL_CONN = os.getenv(
    "PSQL_CONN", "postgresql+asyncpg://postgres:postgres@localhost:5432/phase5_rag"
)


class Base(DeclarativeBase):
    pass


engine = create_async_engine(PSQL_CONN, pool_pre_ping=True, pool_recycle=3600)
session = async_sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


async def db_init():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_db_session():
    async with session() as db:
        yield db
