
import logging
from app.config import get_settings
from elasticsearch import AsyncElasticsearch, NotFoundError

from models.types import Chunk

logger = logging.getLogger(__name__)
settings = get_settings()

INDEX_MAPPING = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0, # prod should have >= 1
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

            # Content for BM25 search
            "content_ltks": {
                "type": "text",
                "analyzer": "whitespace_lowercase",
                "search_analyzer": "whitespace_lowercase"
            },

            "content": {
                "type": "text",
                "index": False
            },

            "title_tks": {
                "type": "text",
                "analyzer": "whitespace_lowercase"
            },
            
            # Vector for semantic search (KNN)
            f"q_{settings.embedding_dimension}_vec": {
                "type": "dense_vector",
                "dims": settings.embedding_dimension,
                "index": True,
                "similarity": "cosine"
            },
            #Metadata
            "chunk_index": {"type": "integer"},
            "created_at": {"type": "date"},
            "metadata": {"type": "object", "enabled": False}

        }
    }
}

class ESIndexer:
    def __init__(self) -> None:
        self.client: AsyncElasticsearch | None = None
        self.index_name = settings.es_index

    async def connect(self) -> None: 
        if self.client is None:
            self.client = AsyncElasticsearch(
                hosts=[settings.es_host],
                request_timeout=30
            )
            logger.info(f'Connected to ES at {settings.es_host}')
    
    async def close(self):
        if self.client:
            await self.client.close()
            self.client = None

    async def create_index(self):
        try:
            await self.client.indices.get(index=self.index_name)
            logger.info(f"Index {self.index_name} already exists")
        except NotFoundError:
            await self.client.indices.create(
                index=self.index_name,
                body=INDEX_MAPPING
            )
            logger.info(f"Created index: {self.index_name}")
        except Exception as e:
            logger.error(f"Error checking/creating index: {e}", exc_info=True)
            raise


    async def index(self, chunk:dict):
        """
        Index a single chunk with both tokens and vectors
        """
        doc: Chunk = {
            "chunk_id": chunk["chunk_id"], # not using .get() bcz it's a mandatory field
            "doc_id": chunk["doc_id"],
            "content_ltks": " ".join(chunk["content_tokens"]),
            "content": chunk["content"],
            "title_tks": chunk.get("title", ""),
            "chunk_index": chunk.get("chunk_index", 0),
            "created_at": chunk.get("created_at"),
            "metadata": chunk.get("metadata", {})
        }

        #Add the vector embeddings array key to doc
        doc[f'q_{settings.embedding_dimension}_vec'] = chunk['embedding']

        try:
            await self.client.index(
                index=self.index_name,
                id=chunk['chunk_id'],
                document=doc
            )
            logger.debug(f"Indexed chunk: {chunk['chunk_id']}")
        except Exception as e:
            logger.error(f"[INDEXER] Error occured of chunk {chunk['chunk_id']} : {e}", exc_info=True) 
    
    async def bulk_index(self, chunks: list[dict]):

        if not chunks:
            return

        bulk_ops = []
        for chunk in chunks:
            doc: Chunk = {
                "chunk_id": chunk["chunk_id"],
                "doc_id": chunk["doc_id"],
                "content_ltks": " ".join(chunk["content_tokens"]),
                "content": chunk["content"],
                "title_tks": chunk.get("title", ""),
                "chunk_index": chunk.get("chunk_index", 0),
                "created_at": chunk.get("created_at"),
                "metadata": chunk.get("metadata", {})
            }
            #Add the vector embeddings array key to doc
            doc[f'q_{settings.embedding_dimension}_vec'] = chunk['embedding']
            
            #NDJSON approach -> followed by ES
            bulk_ops.append(
                {"index": {"_index": self.index_name, "_id": chunk['chunk_id']}}
            )
            bulk_ops.append(doc)
        
        try:
            await self.client.bulk(operations=bulk_ops)
            logger.info(f"Bulk indexed {len(chunks)} chunks")
        except Exception as e:
            logger.info(f'[INDEXER] Bulk Indexing failed : {e}', exc_info=True)

    async def delete_doc_by_id(self, doc_id: str):
        await self.client.delete_by_query(
            index=self.index_name,
            body={
                "query": {
                    "term": {
                        "doc_id": doc_id
                    }
                }
            }
        )
        logger.info(f"Deleted chunks for doc_id: {doc_id}")

    async def get_stats(self) -> dict:
        stats = await self.client.indices.stats(index=self.index_name)
        return {
             "total_docs": stats["_all"]["primaries"]["docs"]["count"],
            "size_bytes": stats["_all"]["primaries"]["store"]["size_in_bytes"]
        }
