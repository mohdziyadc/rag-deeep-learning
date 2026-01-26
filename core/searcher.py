

from elasticsearch import AsyncElasticsearch
from elasticsearch.dsl import Search, Q
from elasticsearch.dsl.query import Knn
from app.config import get_settings
from core.embedder import embedder
from core.tokenizer import Tokenizer
import numpy as np
from models.dtos import SearchResult
import logging

logger = logging.getLogger(__name__)
settings = get_settings()


class HybridSearcher:

    def __init__(self) -> None:
        self.client: AsyncElasticsearch | None = None
        self.indexname = settings.es_index
        self.tokenizer = Tokenizer()

        self.bm25_weight = settings.bm25_weight
        self.vector_weight = settings.vector_weight
        self.rrf_k = 60  # Typical RRF constant

    
    async def connect(self):
        if not self.client or self.client is None:
            self.client = AsyncElasticsearch(
                hosts=[settings.es_host],
                request_timeout=30
            )
            logger.info(f"Searcher Connected!")
            

    async def close(self):
        if self.client:
            await self.client.close()
            self.client = None

    async def search(
        self, 
        query:str,
        top_k: int = 10,
        similarity_threshold: float = 0.2
    ) -> tuple[list[SearchResult], list[float]]:

        # for BM25
        query_tokens = self.tokenizer.tokenize_query(query)
        query_text = " ".join(query_tokens)
        
        # for KNN
        embedder.load()
        query_vector, _ = embedder.encode_query(query)

        bm25_results = await self._bm25_search(
            query_text=query_text,
            top_k=top_k * 2
        )

        knn_results = await self._knn_search(
            query_vector=query_vector.tolist(),
            top_k=top_k*2
        )
        

        results = self.fuse_results(
            bm25_results=bm25_results,
            knn_results=knn_results,
            query_vector=query_vector, 
            similarity_threshold=similarity_threshold, 
            method="rrf"
        )

        return results[:top_k], query_vector.tolist()

    async def _bm25_search(
        self,
        query_text:str,
        top_k: int
    ) -> dict:

        search = Search(using=self.client, index=self.indexname)

        content_match_qry = Q("match", content_ltks={"query": query_text})
        title_match_qry = Q("match", title_tks={"query": query_text, "boost": 2.0})

        bool_qry = Q("bool", should=[
            content_match_qry,
            title_match_qry
        ])

        search = search.query(bool_qry).extra(size=top_k).source(
            includes=["chunk_id", "doc_id", "content", "title_tks", "chunk_index"]
        )

        result = await self.client.search(index=self.indexname, body=search.to_dict())
        return result

    async def _knn_search(self, query_vector: list[float], top_k:int) -> dict:
        vector_field = f"q_{settings.embedding_dimension}_vec"

        search = Search(using=self.client, index=self.indexname)

        knn_query = Knn(
            field=vector_field,
            query_vector=query_vector,
            k=top_k,
            num_candidates=top_k*10
        )

        search = search.knn(knn_query).extra(size=top_k).source(
            includes=["chunk_id", "doc_id", "content", "title_tks", "chunk_index", vector_field]
        )

        result = await self.client.search(index=self.indexname, body=search.to_dict())

        return result
    
    def fuse_results(
        self,
        bm25_results,
        knn_results,
        query_vector: np.ndarray,
        similarity_threshold: float,
        method: str
    ) -> list[SearchResult]:

        '''
        0. calculate hits_to_scores and hits_to_sources map for 
            bm25 and knn results
        1. fused_scores map creation
            if method = rrf:
                do rrf with bm25_scores and knn_scores
            else:
                do normalisation of bm25_scores and knn_scores
                build a fused scores map with chunk_id and calculated 
                fusion scores
        
        2. For each (chunk_id, fused_score) in fused_scores map
            a. get the document vectors from ES (it would be empty [] for BM25 results)
            b. calculate cosine similarity b/w the query vectors and the 
                doc vectors.
            c. if cosine < similarity_threshold, go to the next item
            d. if not, append it to the results as a SearchResult
        3. Sort the results by score in desc
        '''

        vector_field = f"q_{settings.embedding_dimension}_vec"

        def hits_to_scores(results: dict) -> dict[str, float]:
            hits = results.get("hits", {}).get("hits", [])
            return { h["_source"]["chunk_id"]: float(h["_score"]) for h in hits }
        
        def hits_to_sources(results: dict) -> dict[str, dict]:
            hits = results.get("hits", {}).get("hits", [])
            return { h["_source"]["chunk_id"]: h["_source"] for h in hits }

        bm25_scores = hits_to_scores(bm25_results)
        knn_scores = hits_to_scores(knn_results)
        all_sources = {**hits_to_sources(bm25_results), **hits_to_sources(knn_results)}

        if method == "rrf":
            fused_scores = self.rrf_fusion(bm25_scores, knn_scores)
        else:
            bm25_norm = self.minmax_normalize(bm25_scores)
            knn_norm = self.minmax_normalize(knn_scores)
            all_chunk_ids = set(bm25_norm) | set(knn_norm)
            fused_scores = {
                chunk_id: (self.bm25_weight * bm25_norm.get(chunk_id, 0.0))
                        + (self.vector_weight * knn_norm.get(chunk_id, 0.0))
                for chunk_id in all_chunk_ids
            }

        
        results: list[SearchResult] = []
        for chunk_id, fused_score in fused_scores.items():
            source = all_sources.get(chunk_id, {})
            doc_vector = source.get(vector_field, [])

            if doc_vector:
                cosine = self._cosine_similarity(query_vector, np.array(doc_vector))
            else:
                cosine = 0.0

            if cosine < similarity_threshold:
                continue
            
            results.append(SearchResult(
                chunk_id=chunk_id,
                doc_id=source.get("doc_id", ""),
                content=source.get("content", ""),
                title=source.get("title_tks", ""),
                score=float(fused_score),
                bm25_score=float(bm25_scores.get(chunk_id, 0.0)),
                vector_score=float(cosine)
            ))

        results.sort(key=lambda x:x.score, reverse=True)
        return results          

    def rrf_fusion(
        self, 
        bm25_scores: dict[str, float],
        knn_scores: dict[str, float]
    ) -> dict[str, float]:

        def to_rank_map(scores: dict[str, float]) -> dict[str, int]:
            ranked = sorted(scores.items(), key=lambda x:x[1], reverse=True)
            return {doc_id: rank + 1 for rank, (doc_id, _) in enumerate(ranked)}
        
        bm25_ranks = to_rank_map(bm25_scores)
        knn_ranks = to_rank_map(knn_scores)

        # set on a dict only takes the keys
        # from the dict

        #set(bm25_ranks)  = {"chunk_C", "chunk_A", "chunk_D", "chunk_B"}
        #set(knn_ranks)   = {"chunk_B", "chunk_E", "chunk_A", "chunk_C"}
        # all_ids = {"chunk_A", "chunk_B", "chunk_C", "chunk_D", "chunk_E"}

        all_chunk_ids = set(bm25_ranks) | set(knn_ranks)

        fused = {}
        for chunk_id in all_chunk_ids:
            rrf_score = 0.0
            if chunk_id in bm25_ranks:
                rrf_score += 1.0 / (self.rrf_k + bm25_ranks[chunk_id])
            
            if chunk_id in knn_ranks:
                rrf_score += 1.0 / (self.rrf_k + knn_ranks[chunk_id])
            
            fused[chunk_id] = rrf_score
        return fused

    @staticmethod
    def minmax_normalize(scores: dict[str, float]) -> dict[str, float]:
        if not scores:
            return {}
        
        values = list(scores.values())
        min_score, max_score = min(values), max(values)

        if max_score == min_score:
            return { key: 1.0 for key in scores }
        return { key: (value - min_score)/(max_score - min_score) for key, value in scores.items() }

    @staticmethod
    def _cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        dot_product = np.dot(vec_a, vec_b)
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot_product/(norm_a * norm_b))
        





searcher = HybridSearcher()



'''
When you call await self.client.search(...), you get back a dict like this:
{
    "took": 15,                    # Time in milliseconds
    "timed_out": False,
    "_shards": {
        "total": 1,
        "successful": 1,
        "skipped": 0,
        "failed": 0
    },
    "hits": {
        "total": {
            "value": 42,           # Total matching documents
            "relation": "eq"       # "eq" = exact, "gte" = at least
        },
        "max_score": 12.5,         # Highest score
        "hits": [                  # Array of results
            {
                "_index": "your_index",
                "_id": "abc123",
                "_score": 12.5,    # BM25 relevance score
                "_source": {       # Your document fields
                    "chunk_id": "doc1_chunk_0",
                    "doc_id": "doc1",
                    "content": "This is the chunk text...",
                    "title_tks": "Document Title",
                    "chunk_index": 0
                }
            },
            {
                "_index": "your_index",
                "_id": "def456",
                "_score": 10.2,
                "_source": {
                    "chunk_id": "doc2_chunk_3",
                    "doc_id": "doc2",
                    "content": "Another chunk...",
                    "title_tks": "Another Doc",
                    "chunk_index": 3
                }
            }
            // ... more hits
        ]
    }
}
'''