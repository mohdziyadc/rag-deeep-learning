# Mock searcher class
from models.dtos import RetrievedChunk


class Searcher:
    def __init__(self, index_name) -> None:
        self.index_name = index_name

    async def search(self, query: str, top_k: int) -> tuple[list[RetrievedChunk], None]:
        results = [
            RetrievedChunk(
                chunk_id="1",
                doc_id="2",
                content=f"{query}",
                title=f"{self.index_name} - {query}",
                metadata={},
            )
        ]
        return results, None


searcher = Searcher("bruh")


class RetrievalBridge:
    def __init__(self) -> None:
        self.searcher = searcher

    async def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        results, _ = await self.searcher.search(query, top_k=top_k)
        return [
            RetrievedChunk(
                chunk_id=r.chunk_id,
                doc_id=r.doc_id,
                content=r.content,
                title=r.title,
                metadata={},
            )
            for r in results
        ]
