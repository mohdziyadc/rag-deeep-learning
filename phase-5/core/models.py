from datetime import datetime, timezone
from sqlalchemy import BIGINT, DateTime, Float, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column
from core.db import Base


class MemoryModel(Base):
    __tablename__ = "memory"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    tenant_id: Mapped[str] = mapped_column(String(32), index=True, nullable=False)
    memory_type: Mapped[int] = mapped_column(Integer, nullable=False)
    storage_type: Mapped[str] = mapped_column(String(), default="table")
    embd_model: Mapped[str] = mapped_column(String(), nullable=False)
    llm_model: Mapped[str] = mapped_column(String(), nullable=False)
    access_type: Mapped[str] = mapped_column(String(), default="PRIVATE")
    memory_size: Mapped[int] = mapped_column(BIGINT, default=5 * 1024 * 1024)
    foregetting_policy: Mapped[str] = mapped_column(String(), default="FIFO")
    temperature: Mapped[float] = mapped_column(Float, default=0.3)
    system_prompt: Mapped[str] = mapped_column(Text, default="")
    user_prompt: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.now(timezone.utc)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.now(timezone.utc)
    )


class TaskModel(Base):
    __tablename__ = "task"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    memory_id: Mapped[str] = mapped_column(
        String(32), index=True, nullable=False
    )  # memory_id
    task_type: Mapped[str] = mapped_column(String(16), default="MEMORY")
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    progress_msg: Mapped[str] = mapped_column(Text, default="")
    digest: Mapped[str] = mapped_column(
        String(64), default=""
    )  # source raw message_id (reference to raw msg indexed in ES)
    begin_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.now(timezone.utc)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.now(timezone.utc)
    )
