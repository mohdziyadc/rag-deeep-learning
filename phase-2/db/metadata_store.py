from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from sqlalchemy import select
from models.metadata import Base, File, Document

class MetadataStore:

    def __init__(self, db_url: str) -> None:
        self.engine = create_async_engine(db_url, echo=True)
        self.session_factory = async_sessionmaker(self.engine, expire_on_commit=False) 

    async def db_init(self) -> None:
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)


    async def create_file(self, file: File):
        async with self.session_factory() as session:
            session.add(file)
            await session.commit()


    async def create_document(self, doc: Document):
        async with self.session_factory() as session:
            session.add(doc)
            await session.commit()       

    async def update_document(self, doc_id: str, **kwargs):
        async with self.session_factory() as session:
            doc = await session.get(Document, doc_id)
            for k, v in kwargs.items():
                setattr(doc, k, v)
            await session.commit()

    async def get_document(self, doc_id: str) -> Document | None:
        async with self.session_factory() as session:
            return await session.get(Document, doc_id)
    
    async def list_documents(self) -> list[Document]:
        async with self.session_factory() as session:
            result = await session.execute(select(Document).order_by(Document.created_at.desc()))
            return list(result.scalars().all())