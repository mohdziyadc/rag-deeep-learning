from elasticsearch import AsyncElasticsearch


INDEX_MAP = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0,
        "analysis": {
            "analyzer": {
                "whitespace_lowercase": {
                    "type": "custom",
                    "tokenizer": "whitespace",
                    "filter": ["lowercase"]
                }
            }
        }
    },
    "mappings": {
        "properties": {
            "chunk_id": {"type": "keyword"},
            "doc_id": {"type": "keyword"},
            "source_name": {"type": "keyword"},
            "file_type": {"type": "keyword"},
            "section_type": {"type": "keyword"},
            "content_format": {"type": "keyword"},
            "title": {"type": "text", "analyzer": "whitespace_lowercase"},
            "page": {"type": "integer"},
            "content": {"type": "text"},
            "content_ltks": {
                "type": "text",
                "analyzer": "whitespace_lowercase",
                "search_analyzer": "whitespace_lowercase"
            },
            "metadata": {"type": "object", "enabled": False}
        }
    }
}


class ParsedIndex:

    def __init__(self, es_host:str, index_name: str) -> None:
        self.client = AsyncElasticsearch(hosts=[es_host])
        self.indexname = index_name

    async def create_index(self):
        index = await self.client.indices.exists(index=self.indexname)
        if not index:
            await self.client.indices.create(index=self.indexname, body=INDEX_MAP)
    
    async def bulk_insert(self, chunks: list[dict]):
        operations = []

        for chunk in chunks:
            operations.append({"_index": self.indexname, "_id": chunk["chunk_id"]})
            operations.append(chunk)
        
        await self.client.bulk(operations=operations)

    