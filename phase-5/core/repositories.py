from datetime import datetime, timezone
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession


from core.models import MemoryModel


class MemoryRepository:

    def __init__(self, db_session: AsyncSession) -> None:
        self.db = db_session

    async def get(self, memory_id: str):
        return await self.db.get(MemoryModel, memory_id)

    async def list_by_tenant(self, tenant_id: str, page: int = 1, page_size: int = 20):
        # inline indentation by wrapping it in () bcz python
        # complains if u take it to the next line
        query = (
            select(MemoryModel)
            .where(MemoryModel.tenant_id == tenant_id)
            .order_by(MemoryModel.created_at.desc())
            .offset((page - 1) * page_size)
            .limit(page_size)
        )

        """
        Mental model of using .scalars():

        .execute().fetchall() will return Row(id="mem_abc", name="RFP", ...)
        and 
        .scalars().all() will return [{id="mem_abc", name="RFP", ...}]

        .fetchall() and .all() can be replaced by returning list(result)
        """
        result = await self.db.scalars(query)
        return list[MemoryModel](result)

    async def list_by_ids(self, memory_ids: list[str]):
        if not memory_ids:
            return []

        query = select(MemoryModel).where(MemoryModel.id.in_(memory_ids))
        result = await self.db.scalars(query)

        return list(result)

    async def list_all(self):
        result = await self.db.scalars(select(MemoryModel))
        return list(result)

    async def create(self, payload: MemoryModel):
        self.db.add(payload)
        await self.db.commit()
        await self.db.refresh(payload)
        return payload

    async def update(self, memory_id: str, update_dict: dict):
        row = await self.get(memory_id)
        if not row:
            return None
        for k, v in update_dict.items():
            setattr(row, k, v)
        row.update_time = int(datetime.now(timezone.utc).timestamp() * 1000)
        row.update_date = datetime.now(timezone.utc)
        await self.db.commit()
        await self.db.refresh(row)
        return row

    async def delete(self, memory_id: str):
        row = await self.get(memory_id)
        if not row:
            return False
        await self.db.delete(row)
        await self.db.commit()
        return True
