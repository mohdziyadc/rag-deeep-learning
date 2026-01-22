# Phase 1: Building a Mini RAG with Hybrid Search

## 🎯 Goal
Build a simplified hybrid retrieval system that combines **BM25 (keyword search)** + **Vector Search (semantic similarity)** to understand how production RAG systems like RAGFlow work.

---

## 📚 Table of Contents
1. [The Big Picture](#1-the-big-picture)
2. [Understanding the Components](#2-understanding-the-components)
3. [Step-by-Step Implementation](#3-step-by-step-implementation)
4. [Deep Dive: The RAGFlow Code](#4-deep-dive-the-ragflow-code)
5. [Testing Your Implementation](#5-testing-your-implementation)

---

## 1. The Big Picture

### What is Hybrid Search?

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           HYBRID SEARCH                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│    User Query: "How does photosynthesis work in plants?"                     │
│                           │                                                  │
│                           ▼                                                  │
│              ┌────────────┴────────────┐                                     │
│              │                         │                                     │
│              ▼                         ▼                                     │
│    ┌─────────────────┐      ┌─────────────────┐                              │
│    │   BM25 Search   │      │  Vector Search  │                              │
│    │   (Keywords)    │      │  (Semantics)    │                              │
│    └────────┬────────┘      └────────┬────────┘                              │
│             │                        │                                       │
│             │ Finds: "photosynthesis │ Finds: "plants convert                │
│             │ plants chlorophyll"    │ sunlight to energy"                   │
│             │                        │                                       │
│             └──────────┬─────────────┘                                       │
│                        │                                                     │
│                        ▼                                                     │
│              ┌─────────────────┐                                             │
│              │  Weighted Fusion │  (5% BM25 + 95% Vector)                    │
│              └────────┬────────┘                                             │
│                       │                                                      │
│                       ▼                                                      │
│              ┌─────────────────┐                                             │
│              │    Reranking    │  (Cross-encoder scores)                     │
│              └────────┬────────┘                                             │
│                       │                                                      │
│                       ▼                                                      │
│              ┌─────────────────┐                                             │
│              │   Top-K Results │                                             │
│              └─────────────────┘                                             │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Why Hybrid Search?

| Search Type | Strengths | Weaknesses |
|-------------|-----------|------------|
| **BM25 (Keyword)** | ✅ Exact matches, rare terms, proper nouns | ❌ Misses synonyms, paraphrases |
| **Vector (Semantic)** | ✅ Understands meaning, synonyms, concepts | ❌ Can miss exact keywords, numbers |
| **Hybrid** | ✅ Best of both worlds | ⚠️ More complex to tune |

### Real Example

Query: **"Apple stock price"**

- **BM25 alone**: Might return documents about literal apples 🍎
- **Vector alone**: Might return documents about "fruit company valuations" 
- **Hybrid**: Correctly finds Apple Inc. stock information ✅

---

## 2. Understanding the Components

### Component Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        MINI RAG ARCHITECTURE                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐                     │
│  │  Documents  │────▶│  Chunking   │────▶│  Embedding  │                     │
│  │   (Input)   │     │  (Split)    │     │  (Vectors)  │                     │
│  └─────────────┘     └─────────────┘     └──────┬──────┘                     │
│                                                  │                           │
│                                                  ▼                           │
│  ┌───────────────────────────────────────────────────────────────────┐       │
│  │                    ELASTICSEARCH                                  │       │
│  │  ┌─────────────────┐    ┌─────────────────┐                       │       │
│  │  │  content_ltks   │    │   q_768_vec     │                       │       │
│  │  │  (BM25 tokens)  │    │ (dense vector)  │                       │       │
│  │  └─────────────────┘    └─────────────────┘                       │       │
│  └───────────────────────────────────────────────────────────────────┘       │
│                                    │                                         │
│                                    │ Query                                   │
│                                    ▼                                         │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐                     │
│  │   Query     │────▶│   Hybrid    │────▶│   Rerank    │                     │
│  │ Processing  │     │   Search    │     │   Results   │                     │
│  └─────────────┘     └─────────────┘     └──────┬──────┘                     │
│                                                  │                           │
│                                                  ▼                           │
│                                          ┌─────────────┐                     │
│                                          │   LLM       │                     │
│                                          │  Response   │                     │
│                                          └─────────────┘                     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### What Each Component Does

#### 1. **Tokenization (BM25 Prep)**
```
Input:  "Machine learning is transforming healthcare"
Output: ["machine", "learning", "transform", "healthcare"]  # stemmed/normalized
```

**Why?** BM25 needs individual terms to calculate term frequency (TF) and inverse document frequency (IDF).

#### 2. **Embedding (Vector Prep)**
```
Input:  "Machine learning is transforming healthcare"
Output: [0.023, -0.156, 0.892, ..., 0.045]  # 768 or 1024 dimensions
```

**Why?** Vectors capture semantic meaning. Similar concepts have vectors close together in space.

#### 3. **BM25 Scoring**
```
BM25(query, document) = Σ IDF(term) × TF(term, document) × boost_factor
```

**Why?** Rewards documents with query terms, especially rare terms (high IDF).

#### 4. **Cosine Similarity**
```
similarity(vec_a, vec_b) = (vec_a · vec_b) / (||vec_a|| × ||vec_b||)
```

**Why?** Measures angle between vectors. Closer vectors = more similar meaning.

#### 5. **Weighted Fusion**
```
final_score = (0.05 × BM25_score) + (0.95 × vector_score)
```

**Why?** RAGFlow found 5%/95% works well for most cases. You can tune this!

---

## 3. Step-by-Step Implementation

### Project Structure

```
rag-deep-learning/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI entry point
│   ├── config.py            # Configuration
│   └── api/
│       ├── __init__.py
│       └── routes/
│           ├── __init__.py
│           ├── documents.py  # Document ingestion endpoints
│           └── search.py     # Search endpoints
├── core/
│   ├── __init__.py
│   ├── chunker.py           # Document chunking
│   ├── embedder.py          # Embedding generation
│   ├── indexer.py           # Elasticsearch operations
│   ├── searcher.py          # Hybrid search
│   └── reranker.py          # Result reranking
├── models/
│   ├── __init__.py
│   └── schemas.py           # Pydantic models
├── tests/
│   └── test_search.py
├── docs/
│   └── PHASE1_HYBRID_SEARCH_GUIDE.md
├── pyproject.toml
├── docker-compose.yml       # Elasticsearch + PostgreSQL
└── .env
```

---

### Step 0: Environment Setup

#### 0.1 Update Dependencies

```toml
# pyproject.toml
[project]
name = "rag-deep-learning"
version = "0.1.0"
description = "Mini RAG system for learning hybrid search"
readme = "README.md"
requires-python = ">=3.12"
dependencies = [
    # Web Framework
    "fastapi>=0.128.0",
    "uvicorn[standard]>=0.40.0",
    "pydantic>=2.12.5",
    "python-dotenv>=1.2.1",
    
    # Database
    "asyncpg>=0.31.0",
    "sqlalchemy[asyncio]>=2.0.0",
    
    # Elasticsearch
    "elasticsearch[async]>=8.0.0",
    "elasticsearch-dsl>=8.0.0",    # Pythonic query builder
    
    # ML/NLP
    "sentence-transformers>=3.0.0",  # For embeddings
    "numpy>=1.26.0",
    "scikit-learn>=1.5.0",           # For cosine similarity
    
    # Tokenization
    "nltk>=3.8.0",                    # For BM25 tokenization
    
    # Optional: Reranking
    # "torch>=2.0.0",                 # If using cross-encoder reranking
]
```

#### 0.2 Docker Compose for Services

```yaml
# docker-compose.yml
version: '3.8'

services:
  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.12.0
    container_name: rag-elasticsearch
    environment:
      - discovery.type=single-node
      - xpack.security.enabled=false
      - "ES_JAVA_OPTS=-Xms512m -Xmx512m"
    ports:
      - "9200:9200"
    volumes:
      - es_data:/usr/share/elasticsearch/data
    healthcheck:
      test: ["CMD-SHELL", "curl -f http://localhost:9200/_cluster/health || exit 1"]
      interval: 10s
      timeout: 5s
      retries: 5

  postgres:
    image: postgres:16-alpine
    container_name: rag-postgres
    environment:
      POSTGRES_USER: raguser
      POSTGRES_PASSWORD: ragpass
      POSTGRES_DB: ragdb
    ports:
      - "5432:5432"
    volumes:
      - pg_data:/var/lib/postgresql/data

volumes:
  es_data:
  pg_data:
```

#### 0.3 Environment Variables

```bash
# .env
# Elasticsearch
ES_HOST=http://localhost:9200
ES_INDEX=rag_chunks

# PostgreSQL
DATABASE_URL=postgresql+asyncpg://raguser:ragpass@localhost:5432/ragdb

# Embedding Model (using free, local model)
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DIMENSION=384

# Search Config
BM25_WEIGHT=0.05
VECTOR_WEIGHT=0.95
TOP_K=10
```

---

### Step 1: Index Documents into Elasticsearch

**What:** Store documents with both tokenized text (for BM25) and vectors (for semantic search).

**Why:** Elasticsearch supports both full-text search and dense vector search in one index.

**How:**

#### 1.1 Configuration

```python
# app/config.py
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # Elasticsearch
    es_host: str = "http://localhost:9200"
    es_index: str = "rag_chunks"
    
    # PostgreSQL
    database_url: str = "postgresql+asyncpg://raguser:ragpass@localhost:5432/ragdb"
    
    # Embedding
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    
    # Search
    bm25_weight: float = 0.05
    vector_weight: float = 0.95
    top_k: int = 10
    
    class Config:
        env_file = ".env"


@lru_cache
def get_settings() -> Settings:
    return Settings()
```

#### 1.2 Pydantic Models

```python
# models/schemas.py
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class DocumentCreate(BaseModel):
    """Input for creating a document"""
    title: str
    content: str
    metadata: dict = Field(default_factory=dict)


class ChunkInDB(BaseModel):
    """Chunk stored in Elasticsearch"""
    chunk_id: str
    doc_id: str
    content: str
    content_tokens: list[str]  # For BM25
    embedding: list[float]     # For vector search
    title: str
    chunk_index: int
    metadata: dict = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class SearchQuery(BaseModel):
    """Search request"""
    question: str
    top_k: int = 10
    similarity_threshold: float = 0.2
    use_rerank: bool = True


class SearchResult(BaseModel):
    """Single search result"""
    chunk_id: str
    doc_id: str
    content: str
    title: str
    score: float
    bm25_score: float
    vector_score: float


class SearchResponse(BaseModel):
    """Search response"""
    query: str
    total: int
    results: list[SearchResult]
```

#### 1.3 Elasticsearch Index Setup

```python
# core/indexer.py
"""
Elasticsearch Index Management

WHY: We need a place to store documents that supports both:
  1. Full-text search (BM25) - for keyword matching
  2. Vector search (KNN) - for semantic similarity

WHAT: This module handles:
  - Creating the index with proper mappings
  - Indexing documents with tokens and vectors
  - Deleting documents

HOW: Elasticsearch 8.x supports dense_vector fields with HNSW index
"""
from elasticsearch import AsyncElasticsearch
from app.config import get_settings
import logging

logger = logging.getLogger(__name__)
settings = get_settings()


# Index mapping - this defines how data is stored and searched
INDEX_MAPPING = {
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
            # Identifiers
            "chunk_id": {"type": "keyword"},
            "doc_id": {"type": "keyword"},
            
            # Content for BM25 search
            # Using whitespace analyzer to match RAGFlow's approach
            "content_ltks": {
                "type": "text",
                "analyzer": "whitespace_lowercase",
                "search_analyzer": "whitespace_lowercase"
            },
            
            # Raw content for display
            "content": {
                "type": "text",
                "index": False  # Not searchable, just stored
            },
            
            # Title for boosting
            "title_tks": {
                "type": "text",
                "analyzer": "whitespace_lowercase"
            },
            
            # Dense vector for semantic search
            # Dimension must match your embedding model!
            f"q_{settings.embedding_dimension}_vec": {
                "type": "dense_vector",
                "dims": settings.embedding_dimension,
                "index": True,
                "similarity": "cosine"
            },
            
            # Metadata
            "chunk_index": {"type": "integer"},
            "created_at": {"type": "date"},
            "metadata": {"type": "object", "enabled": False}
        }
    }
}


class ESIndexer:
    """
    Handles all Elasticsearch operations for our RAG system.
    
    This is a simplified version of RAGFlow's DocStoreConnection.
    """
    
    def __init__(self):
        self.client: AsyncElasticsearch | None = None
        self.index_name = settings.es_index
    
    async def connect(self):
        """Initialize Elasticsearch connection"""
        if self.client is None:
            self.client = AsyncElasticsearch(
                hosts=[settings.es_host],
                request_timeout=30
            )
            logger.info(f"Connected to Elasticsearch at {settings.es_host}")
    
    async def close(self):
        """Close Elasticsearch connection"""
        if self.client:
            await self.client.close()
            self.client = None
    
    async def create_index(self):
        """
        Create the index with proper mappings.
        
        WHY: The mapping defines how ES stores and searches data.
        - text fields: Full-text search with BM25
        - dense_vector: K-nearest neighbor search
        """
        if not await self.client.indices.exists(index=self.index_name):
            await self.client.indices.create(
                index=self.index_name,
                body=INDEX_MAPPING
            )
            logger.info(f"Created index: {self.index_name}")
        else:
            logger.info(f"Index {self.index_name} already exists")
    
    async def index_chunk(self, chunk: dict):
        """
        Index a single chunk with both tokens and vector.
        
        WHY: Each chunk needs:
        - Tokenized content for BM25 scoring
        - Vector embedding for semantic similarity
        """
        doc = {
            "chunk_id": chunk["chunk_id"],
            "doc_id": chunk["doc_id"],
            "content_ltks": " ".join(chunk["content_tokens"]),  # Tokens joined by space
            "content": chunk["content"],
            "title_tks": chunk.get("title", ""),
            f"q_{settings.embedding_dimension}_vec": chunk["embedding"],
            "chunk_index": chunk.get("chunk_index", 0),
            "created_at": chunk.get("created_at"),
            "metadata": chunk.get("metadata", {})
        }
        
        await self.client.index(
            index=self.index_name,
            id=chunk["chunk_id"],
            document=doc
        )
        logger.debug(f"Indexed chunk: {chunk['chunk_id']}")
    
    async def bulk_index(self, chunks: list[dict]):
        """
        Bulk index multiple chunks for efficiency.
        
        WHY: Bulk operations are much faster than individual inserts.
        """
        if not chunks:
            return
        
        operations = []
        for chunk in chunks:
            operations.append({"index": {"_index": self.index_name, "_id": chunk["chunk_id"]}})
            operations.append({
                "chunk_id": chunk["chunk_id"],
                "doc_id": chunk["doc_id"],
                "content_ltks": " ".join(chunk["content_tokens"]),
                "content": chunk["content"],
                "title_tks": chunk.get("title", ""),
                f"q_{settings.embedding_dimension}_vec": chunk["embedding"],
                "chunk_index": chunk.get("chunk_index", 0),
                "created_at": chunk.get("created_at"),
                "metadata": chunk.get("metadata", {})
            })
        
        await self.client.bulk(operations=operations)
        logger.info(f"Bulk indexed {len(chunks)} chunks")
    
    async def delete_by_doc_id(self, doc_id: str):
        """Delete all chunks for a document"""
        await self.client.delete_by_query(
            index=self.index_name,
            body={"query": {"term": {"doc_id": doc_id}}}
        )
        logger.info(f"Deleted chunks for doc_id: {doc_id}")
    
    async def get_stats(self) -> dict:
        """Get index statistics"""
        stats = await self.client.indices.stats(index=self.index_name)
        return {
            "total_docs": stats["_all"]["primaries"]["docs"]["count"],
            "size_bytes": stats["_all"]["primaries"]["store"]["size_in_bytes"]
        }


# Singleton instance
indexer = ESIndexer()
```

---

### Step 2: Implement BM25 Search

**What:** Search using keyword matching with TF-IDF weighting.

**Why:** BM25 excels at finding exact matches and rare terms. If someone searches for "NVIDIA RTX 4090", we want documents with those exact terms!

**How:**

```python
# core/tokenizer.py
"""
Text Tokenization for BM25

WHY: BM25 works on individual terms. We need to:
  1. Split text into words
  2. Normalize (lowercase, remove punctuation)
  3. Optionally stem (running -> run)

WHAT: This matches RAGFlow's rag_tokenizer approach

HOW: Using NLTK for robust tokenization
"""
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import logging

logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)


class Tokenizer:
    """
    Tokenizer for BM25 search.
    
    This is a simplified version of RAGFlow's rag_tokenizer.
    RAGFlow does more sophisticated things like:
    - Chinese text segmentation (jieba)
    - Fine-grained tokenization for compound words
    - Synonym expansion
    
    We'll keep it simple for learning.
    """
    
    def __init__(self, remove_stopwords: bool = False, use_stemming: bool = False):
        self.remove_stopwords = remove_stopwords
        self.use_stemming = use_stemming
        self.stemmer = PorterStemmer() if use_stemming else None
        self.stop_words = set(stopwords.words('english')) if remove_stopwords else set()
    
    def tokenize(self, text: str) -> list[str]:
        """
        Tokenize text for BM25 indexing/searching.
        
        Steps:
        1. Lowercase
        2. Remove special characters (keep alphanumeric)
        3. Split into words
        4. Optionally remove stopwords
        5. Optionally stem
        """
        # Lowercase and normalize
        text = text.lower()
        
        # Remove special characters but keep spaces
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Split into tokens
        tokens = text.split()
        
        # Remove stopwords
        if self.remove_stopwords:
            tokens = [t for t in tokens if t not in self.stop_words]
        
        # Stem tokens
        if self.use_stemming and self.stemmer:
            tokens = [self.stemmer.stem(t) for t in tokens]
        
        # Filter out very short tokens
        tokens = [t for t in tokens if len(t) > 1]
        
        return tokens
    
    def tokenize_query(self, query: str) -> list[str]:
        """
        Tokenize a search query.
        
        Same as document tokenization for consistency.
        """
        return self.tokenize(query)


# Example usage:
# tokenizer = Tokenizer()
# tokens = tokenizer.tokenize("Machine Learning is transforming Healthcare!")
# # Result: ['machine', 'learning', 'is', 'transforming', 'healthcare']
```

---

### Step 3: Add Vector Search with Embeddings

**What:** Convert text to dense vectors and search by semantic similarity.

**Why:** Vectors understand meaning! "car" and "automobile" have similar vectors even though they're different words.

**How:**

```python
# core/embedder.py
"""
Embedding Generation

WHY: Vector search requires converting text to dense vectors.
  - Similar meanings = vectors close together
  - Enables semantic search beyond keyword matching

WHAT: Using sentence-transformers (free, runs locally!)

HOW: 
  1. Load a pre-trained model
  2. Encode text to vectors
  3. Batch for efficiency

This is a simplified version of RAGFlow's embedding_model.py
"""
from sentence_transformers import SentenceTransformer
import numpy as np
from app.config import get_settings
import logging

logger = logging.getLogger(__name__)
settings = get_settings()


class Embedder:
    """
    Generate embeddings using sentence-transformers.
    
    RAGFlow supports many providers (OpenAI, Cohere, etc.)
    We use a local model for simplicity and cost.
    
    Model options (all free, local):
    - all-MiniLM-L6-v2: Fast, 384 dims, good quality
    - all-mpnet-base-v2: Better quality, 768 dims, slower
    - BAAI/bge-small-en-v1.5: Great for RAG, 384 dims
    """
    
    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.embedding_model
        self.model: SentenceTransformer | None = None
        self.dimension = settings.embedding_dimension
    
    def load(self):
        """Load the embedding model"""
        if self.model is None:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            # Update dimension based on actual model
            self.dimension = self.model.get_sentence_embedding_dimension()
            logger.info(f"Model loaded. Dimension: {self.dimension}")
    
    def encode(self, texts: list[str]) -> tuple[np.ndarray, int]:
        """
        Encode texts to vectors.
        
        Returns:
            - embeddings: numpy array of shape (n_texts, dimension)
            - token_count: approximate token count for tracking
        
        This matches RAGFlow's encode() signature.
        """
        if self.model is None:
            self.load()
        
        # Batch encoding for efficiency
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=True  # Important for cosine similarity!
        )
        
        # Approximate token count (for monitoring)
        token_count = sum(len(t.split()) for t in texts)
        
        return np.array(embeddings), token_count
    
    def encode_query(self, query: str) -> tuple[np.ndarray, int]:
        """
        Encode a single query.
        
        Some models have different encoding for queries vs documents.
        sentence-transformers handles this automatically for symmetric models.
        """
        embeddings, token_count = self.encode([query])
        return embeddings[0], token_count


# Singleton instance
embedder = Embedder()


# Example usage:
# embedder.load()
# vectors, tokens = embedder.encode(["Hello world", "How are you?"])
# print(vectors.shape)  # (2, 384)
```

---

### Step 4: Combine with Weighted Fusion

**What:** Retrieve BM25 and vector results separately, then fuse scores in Python.

**Why:** Elasticsearch does **not** produce a true weighted sum between BM25 and KNN.  
The `boost` on the `knn` block only scales vector scores **within** the KNN retriever, it does not combine with BM25 the way most people expect.  
For a real hybrid fusion, you must **run two searches** and **merge scores manually** (weighted sum or RRF).

**How:**

```python
# core/searcher.py
"""
Hybrid Search Implementation (true fusion)

WHY: Combine BM25 (keyword) + Vector (semantic) search
  - BM25 catches exact matches
  - Vector catches semantic similarity
  - Together = robust retrieval

HOW:
  1. Run BM25 search (text-only)
  2. Run KNN search (vector-only)
  3. Normalize and fuse scores (weighted sum or RRF)
  4. Optionally rerank
"""
from elasticsearch import AsyncElasticsearch
from elasticsearch_dsl import Search, Q
from elasticsearch_dsl.query import Knn
from app.config import get_settings
from core.embedder import embedder
from core.tokenizer import Tokenizer
from models.schemas import SearchResult
import numpy as np
import logging

logger = logging.getLogger(__name__)
settings = get_settings()


class HybridSearcher:
    def __init__(self):
        self.client: AsyncElasticsearch | None = None
        self.index_name = settings.es_index
        self.tokenizer = Tokenizer()
        self.bm25_weight = settings.bm25_weight
        self.vector_weight = settings.vector_weight
        self.rrf_k = 60  # Typical RRF constant

    async def connect(self):
        if self.client is None:
            self.client = AsyncElasticsearch(
                hosts=[settings.es_host],
                request_timeout=30
            )

    async def search(
        self,
        query: str,
        top_k: int = 10,
        similarity_threshold: float = 0.2,
        fusion_method: str = "weighted_sum"  # or "rrf"
    ) -> tuple[list[SearchResult], list[float]]:
        query_tokens = self.tokenizer.tokenize_query(query)
        query_text = " ".join(query_tokens)

        embedder.load()
        query_vector, _ = embedder.encode_query(query)

        # Run two separate searches
        bm25_resp = await self._bm25_search(query_text, top_k * 2)
        knn_resp = await self._knn_search(query_vector.tolist(), top_k * 2)

        # Fuse the scores
        results = self._fuse_results(
            bm25_resp=bm25_resp,
            knn_resp=knn_resp,
            query_vector=query_vector,
            similarity_threshold=similarity_threshold,
            method=fusion_method
        )

        return results[:top_k], query_vector.tolist()

    async def _bm25_search(self, query_text: str, top_k: int) -> dict:
        search = (
            Search(using=self.client, index=self.index_name)
            .query(Q("bool", should=[
                Q("match", content_ltks={"query": query_text}),
                Q("match", title_tks={"query": query_text, "boost": 2.0})
            ]))
            .extra(size=top_k)
            .source(includes=["chunk_id", "doc_id", "content", "title_tks", "chunk_index"])
        )
        return await self.client.search(index=self.index_name, body=search.to_dict())

    async def _knn_search(self, query_vector: list[float], top_k: int) -> dict:
        vector_field = f"q_{settings.embedding_dimension}_vec"
        search = (
            Search(using=self.client, index=self.index_name)
            .knn(Knn(
                field=vector_field,
                query_vector=query_vector,
                k=top_k,
                num_candidates=top_k * 10
            ))
            .extra(size=top_k)
            .source(includes=["chunk_id", "doc_id", "content", "title_tks", "chunk_index", vector_field])
        )
        return await self.client.search(index=self.index_name, body=search.to_dict())

    def _fuse_results(
        self,
        bm25_resp: dict,
        knn_resp: dict,
        query_vector: np.ndarray,
        similarity_threshold: float,
        method: str
    ) -> list[SearchResult]:
        vector_field = f"q_{settings.embedding_dimension}_vec"

        def hits_to_scores(resp: dict) -> dict[str, float]:
            hits = resp.get("hits", {}).get("hits", [])
            return {h["_source"]["chunk_id"]: float(h["_score"]) for h in hits}

        def hits_to_sources(resp: dict) -> dict[str, dict]:
            hits = resp.get("hits", {}).get("hits", [])
            return {h["_source"]["chunk_id"]: h["_source"] for h in hits}

        bm25_scores = hits_to_scores(bm25_resp)
        knn_scores = hits_to_scores(knn_resp)
        sources = {**hits_to_sources(bm25_resp), **hits_to_sources(knn_resp)}

        if method == "rrf":
            fused_scores = self._rrf_fusion(bm25_scores, knn_scores)
        else:
            bm25_norm = self._minmax_normalize(bm25_scores)
            knn_norm = self._minmax_normalize(knn_scores)
            all_ids = set(bm25_norm) | set(knn_norm)
            fused_scores = {
                cid: (self.bm25_weight * bm25_norm.get(cid, 0.0)) +
                     (self.vector_weight * knn_norm.get(cid, 0.0))
                for cid in all_ids
            }

        results = []
        for chunk_id, fused_score in fused_scores.items():
            source = sources.get(chunk_id, {})
            doc_vector = source.get(vector_field, [])
            vector_score = self._cosine_similarity(query_vector, np.array(doc_vector)) if doc_vector else 0.0
            if vector_score < similarity_threshold:
                continue
            results.append(SearchResult(
                chunk_id=chunk_id,
                doc_id=source.get("doc_id", ""),
                content=source.get("content", ""),
                title=source.get("title_tks", ""),
                score=float(fused_score),
                bm25_score=float(bm25_scores.get(chunk_id, 0.0)),
                vector_score=float(vector_score)
            ))

        results.sort(key=lambda x: x.score, reverse=True)
        return results

    def _rrf_fusion(self, bm25_scores: dict[str, float], knn_scores: dict[str, float]) -> dict[str, float]:
        def to_rank_map(scores: dict[str, float]) -> dict[str, int]:
            ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            return {doc_id: rank + 1 for rank, (doc_id, _) in enumerate(ranked)}

        bm25_ranks = to_rank_map(bm25_scores)
        knn_ranks = to_rank_map(knn_scores)
        all_ids = set(bm25_ranks) | set(knn_ranks)

        fused = {}
        for cid in all_ids:
            rrf_score = 0.0
            if cid in bm25_ranks:
                rrf_score += 1.0 / (self.rrf_k + bm25_ranks[cid])
            if cid in knn_ranks:
                rrf_score += 1.0 / (self.rrf_k + knn_ranks[cid])
            fused[cid] = rrf_score
        return fused

    @staticmethod
    def _minmax_normalize(scores: dict[str, float]) -> dict[str, float]:
        if not scores:
            return {}
        values = list(scores.values())
        min_s = min(values)
        max_s = max(values)
        if max_s == min_s:
            return {k: 1.0 for k in scores}
        return {k: (v - min_s) / (max_s - min_s) for k, v in scores.items()}

    @staticmethod
    def _cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        dot_product = np.dot(vec_a, vec_b)
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot_product / (norm_a * norm_b))
```

#### Understanding elasticsearch-dsl Query Building

The key benefit of `elasticsearch-dsl` is its Pythonic interface for building queries:

**Raw dict approach (old):**
```python
{
    "query": {
        "bool": {
            "should": [
                {"match": {"content_ltks": {"query": text, "boost": 0.05}}}
            ]
        }
    }
}
```

**elasticsearch-dsl approach (new):**
```python
from elasticsearch_dsl import Search, Q

content_match = Q('match', content_ltks={'query': text, 'boost': 0.05})
search = Search(using=client, index=index_name).query(Q('bool', should=[content_match]))
```

**Key elasticsearch-dsl classes:**

| Class | Purpose | Example |
|-------|---------|---------|
| `Search` | Main search builder | `Search(using=client, index=index_name).query(q)` |
| `Q` | Query factory | `Q('match', field={'query': text})` |
| `Knn` | KNN vector query | `Knn(field='vec', query_vector=[...])` |

**Chaining methods:**
```python
search = (
    Search(using=client, index=index_name)
    .query(bool_query)           # BM25-only query
    .source(includes=['field'])  # Select fields
    .extra(size=10)              # Set size
    .highlight('content')        # Add highlighting
)

# Convert to dict for raw client
search.to_dict()
```

---

### Step 5: Add Reranking

**What:** Use a more powerful model to rescore the top results.

**Why:** Initial retrieval is fast but rough. Reranking is slower but more accurate.

**How:**

```python
# core/reranker.py
"""
Result Reranking

WHY: Two-stage retrieval is more effective:
  1. Fast retrieval: Get top-100 candidates quickly
  2. Slow reranking: Score top candidates more accurately

WHAT: RAGFlow supports multiple rerankers:
  - Cross-encoders (BERT-based)
  - BGE reranker
  - Cohere rerank API

HOW: Cross-encoders see (query, document) pairs together,
     unlike bi-encoders which encode them separately.

For this mini RAG, we'll use a simple token similarity reranker
(like RAGFlow's fallback) to avoid heavy dependencies.
"""
from core.tokenizer import Tokenizer
from models.schemas import SearchResult
from app.config import get_settings
import numpy as np
from collections import Counter
import logging

logger = logging.getLogger(__name__)
settings = get_settings()


class TokenReranker:
    """
    Simple token-based reranker.
    
    This is similar to RAGFlow's rerank() method in search.py
    which uses hybrid_similarity when no rerank model is available.
    
    For production, use a cross-encoder model like:
    - cross-encoder/ms-marco-MiniLM-L-6-v2
    - BAAI/bge-reranker-base
    """
    
    def __init__(self):
        self.tokenizer = Tokenizer()
    
    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        query_vector: list[float],
        tkweight: float = 0.3,
        vtweight: float = 0.7
    ) -> list[SearchResult]:
        """
        Rerank results using hybrid similarity.
        
        This matches RAGFlow's rerank() method:
        - Token similarity: Jaccard-like overlap
        - Vector similarity: Cosine similarity
        - Combined with weights
        
        Args:
            query: Original query string
            results: Results from initial retrieval
            query_vector: Query embedding
            tkweight: Weight for token similarity
            vtweight: Weight for vector similarity
        
        Returns:
            Reranked list of results
        """
        if not results:
            return results
        
        # Tokenize query
        query_tokens = self.tokenizer.tokenize_query(query)
        
        # Compute token similarity for each result
        token_scores = []
        for result in results:
            doc_tokens = self.tokenizer.tokenize(result.content)
            tk_sim = self._token_similarity(query_tokens, doc_tokens)
            token_scores.append(tk_sim)
        
        # Normalize vector scores (already computed)
        vector_scores = [r.vector_score for r in results]
        
        # Combine scores
        for i, result in enumerate(results):
            combined = (tkweight * token_scores[i]) + (vtweight * vector_scores[i])
            result.score = combined
            result.bm25_score = token_scores[i]  # Update for transparency
        
        # Sort by new score
        results.sort(key=lambda x: x.score, reverse=True)
        
        return results
    
    def _token_similarity(self, query_tokens: list[str], doc_tokens: list[str]) -> float:
        """
        Compute token overlap similarity.
        
        This is a simplified version of RAGFlow's similarity() in query.py
        which uses term weights (TF-IDF).
        
        We use a simple weighted overlap:
        - Count matching tokens
        - Weight by position (earlier = more important)
        """
        if not query_tokens or not doc_tokens:
            return 0.0
        
        query_set = set(query_tokens)
        doc_counter = Counter(doc_tokens)
        
        # Sum up weights for matching tokens
        score = 0.0
        for token in query_tokens:
            if token in doc_counter:
                score += 1.0  # Simple binary match
        
        # Normalize by query length
        return score / len(query_tokens)


class CrossEncoderReranker:
    """
    Cross-encoder reranker using sentence-transformers.
    
    OPTIONAL: Uncomment and install dependencies if you want
    more accurate reranking.
    
    Usage:
        reranker = CrossEncoderReranker()
        reranker.load()
        results = reranker.rerank(query, results)
    """
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self.model = None
    
    def load(self):
        """Load the cross-encoder model"""
        from sentence_transformers import CrossEncoder
        if self.model is None:
            logger.info(f"Loading cross-encoder: {self.model_name}")
            self.model = CrossEncoder(self.model_name)
    
    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        **kwargs
    ) -> list[SearchResult]:
        """
        Rerank using cross-encoder scores.
        
        Cross-encoders are more accurate because they see
        query and document together, capturing interactions.
        """
        if not results or self.model is None:
            return results
        
        # Prepare pairs for cross-encoder
        pairs = [(query, r.content) for r in results]
        
        # Get scores
        scores = self.model.predict(pairs)
        
        # Update results
        for i, result in enumerate(results):
            result.score = float(scores[i])
        
        # Sort by new score
        results.sort(key=lambda x: x.score, reverse=True)
        
        return results


# Default reranker (token-based, no extra dependencies)
reranker = TokenReranker()
```

---

### Step 6: Document Chunking

**What:** Split documents into smaller pieces for indexing.

**Why:** 
- LLMs have context limits
- Smaller chunks = more precise retrieval
- Better citation granularity

**How:**

```python
# core/chunker.py
"""
Document Chunking

WHY: We need to split documents into smaller pieces because:
  1. Embedding models have token limits
  2. Smaller chunks = more precise retrieval
  3. LLM context windows are limited

WHAT: RAGFlow has multiple chunking strategies:
  - Naive: Fixed-size chunks with overlap
  - Paper: By sections (abstract, introduction, etc.)
  - QA: Question-answer pairs
  - Book: By chapters

HOW: We'll implement naive chunking for simplicity.
"""
from models.schemas import ChunkInDB
from core.tokenizer import Tokenizer
from core.embedder import embedder
from datetime import datetime
import uuid
import logging

logger = logging.getLogger(__name__)


class NaiveChunker:
    """
    Simple fixed-size chunking with overlap.
    
    This is the most common chunking strategy.
    RAGFlow's implementation is in rag/app/naive.py
    """
    
    def __init__(
        self,
        chunk_size: int = 512,      # Characters per chunk
        chunk_overlap: int = 50,     # Overlap between chunks
        min_chunk_size: int = 100    # Minimum chunk size
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.tokenizer = Tokenizer()
    
    def chunk_document(
        self,
        doc_id: str,
        title: str,
        content: str,
        metadata: dict = None
    ) -> list[dict]:
        """
        Split a document into chunks.
        
        Args:
            doc_id: Unique document identifier
            title: Document title
            content: Full document text
            metadata: Additional metadata
        
        Returns:
            List of chunk dictionaries ready for indexing
        """
        chunks = []
        
        # Split into chunks with overlap
        start = 0
        chunk_index = 0
        
        while start < len(content):
            # Find the end of this chunk
            end = start + self.chunk_size
            
            # Try to break at a sentence or paragraph boundary
            if end < len(content):
                # Look for sentence end
                for punct in ['. ', '! ', '? ', '\n\n', '\n']:
                    pos = content.rfind(punct, start, end)
                    if pos > start + self.min_chunk_size:
                        end = pos + len(punct)
                        break
            
            chunk_text = content[start:end].strip()
            
            if len(chunk_text) >= self.min_chunk_size:
                chunks.append({
                    "chunk_id": str(uuid.uuid4()),
                    "doc_id": doc_id,
                    "content": chunk_text,
                    "content_tokens": self.tokenizer.tokenize(chunk_text),
                    "title": title,
                    "chunk_index": chunk_index,
                    "metadata": metadata or {},
                    "created_at": datetime.utcnow().isoformat()
                })
                chunk_index += 1
            
            # Move start with overlap
            start = end - self.chunk_overlap
        
        logger.info(f"Created {len(chunks)} chunks from document {doc_id}")
        return chunks
    
    async def chunk_and_embed(
        self,
        doc_id: str,
        title: str,
        content: str,
        metadata: dict = None
    ) -> list[dict]:
        """
        Chunk document and generate embeddings.
        
        This combines chunking and embedding for efficiency.
        """
        # First, create chunks
        chunks = self.chunk_document(doc_id, title, content, metadata)
        
        if not chunks:
            return chunks
        
        # Then, batch embed all chunks
        embedder.load()
        chunk_texts = [c["content"] for c in chunks]
        embeddings, _ = embedder.encode(chunk_texts)
        
        # Add embeddings to chunks
        for i, chunk in enumerate(chunks):
            chunk["embedding"] = embeddings[i].tolist()
        
        return chunks


# Default chunker
chunker = NaiveChunker()
```

---

### Step 7: FastAPI Application

**What:** REST API for document ingestion and search.

**Why:** Provides a clean interface to interact with the RAG system.

**How:**

```python
# app/main.py
"""
FastAPI Application

The main entry point for our mini RAG system.
"""
from fastapi import FastAPI
from contextlib import asynccontextmanager
from core.indexer import indexer
from core.searcher import searcher
from core.embedder import embedder
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Setup and teardown"""
    # Startup
    logger.info("Starting up Mini RAG...")
    await indexer.connect()
    await indexer.create_index()
    await searcher.connect()
    embedder.load()
    logger.info("Mini RAG ready!")
    
    yield
    
    # Shutdown
    logger.info("Shutting down...")
    await indexer.close()
    await searcher.close()


app = FastAPI(
    title="Mini RAG",
    description="A simplified RAG system for learning hybrid search",
    version="0.1.0",
    lifespan=lifespan
)


# Import routes
from app.api.routes import documents, search

app.include_router(documents.router, prefix="/api/documents", tags=["Documents"])
app.include_router(search.router, prefix="/api/search", tags=["Search"])


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy"}
```

```python
# app/api/routes/documents.py
"""
Document API Routes
"""
from fastapi import APIRouter, HTTPException
from models.schemas import DocumentCreate
from core.chunker import chunker
from core.indexer import indexer
import uuid
import logging

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/")
async def ingest_document(document: DocumentCreate):
    """
    Ingest a document into the RAG system.
    
    Steps:
    1. Generate unique doc_id
    2. Chunk the document
    3. Generate embeddings
    4. Index into Elasticsearch
    """
    doc_id = str(uuid.uuid4())
    
    try:
        # Chunk and embed
        chunks = await chunker.chunk_and_embed(
            doc_id=doc_id,
            title=document.title,
            content=document.content,
            metadata=document.metadata
        )
        
        if not chunks:
            raise HTTPException(status_code=400, detail="Document too short to chunk")
        
        # Index chunks
        await indexer.bulk_index(chunks)
        
        return {
            "doc_id": doc_id,
            "title": document.title,
            "chunks_created": len(chunks)
        }
    
    except Exception as e:
        logger.error(f"Error ingesting document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{doc_id}")
async def delete_document(doc_id: str):
    """Delete a document and all its chunks"""
    try:
        await indexer.delete_by_doc_id(doc_id)
        return {"message": f"Document {doc_id} deleted"}
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_stats():
    """Get index statistics"""
    try:
        stats = await indexer.get_stats()
        return stats
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))
```

```python
# app/api/routes/search.py
"""
Search API Routes
"""
from fastapi import APIRouter, HTTPException
from models.schemas import SearchQuery, SearchResponse
from core.searcher import searcher
from core.reranker import reranker
import logging

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/", response_model=SearchResponse)
async def hybrid_search(query: SearchQuery):
    """
    Perform hybrid search.
    
    This demonstrates the full RAG retrieval pipeline:
    1. Process query (tokenize + embed)
    2. Hybrid search (BM25 + vector)
    3. Optional reranking
    4. Return top-k results
    """
    try:
        # Perform hybrid search
        results, query_vector = await searcher.search(
            query=query.question,
            top_k=query.top_k * 2 if query.use_rerank else query.top_k,
            similarity_threshold=query.similarity_threshold
        )
        
        # Rerank if requested
        if query.use_rerank and results:
            results = reranker.rerank(
                query=query.question,
                results=results,
                query_vector=query_vector
            )
            results = results[:query.top_k]
        
        return SearchResponse(
            query=query.question,
            total=len(results),
            results=results
        )
    
    except Exception as e:
        logger.error(f"Error searching: {e}")
        raise HTTPException(status_code=500, detail=str(e))
```

```python
# app/api/routes/__init__.py
from . import documents, search
```

```python
# app/api/__init__.py
```

```python
# app/__init__.py
```

```python
# models/__init__.py
```

```python
# core/__init__.py
```

---

## 4. Deep Dive: The RAGFlow Code

Now let's map our simplified code back to RAGFlow's production implementation.

### 4.1 RAGFlow's Dealer Class (search.py)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     RAGFlow Dealer Class Breakdown                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  class Dealer:                                                               │
│      │                                                                       │
│      ├── __init__(dataStore)                                                 │
│      │   └── Our ESIndexer + HybridSearcher combined                         │
│      │                                                                       │
│      ├── search()           # Lines 75-172                                   │
│      │   ├── get_filters()  # Build ES filters for kb_ids, doc_ids          │
│      │   ├── get_vector()   # Encode query with embedding model              │
│      │   ├── qryr.question()# Tokenize and build BM25 query                  │
│      │   └── dataStore.search() # Execute hybrid search                      │
│      │                                                                       │
│      ├── retrieval()        # Lines 363-511 - THE MAIN METHOD                │
│      │   ├── search()       # Get initial candidates                         │
│      │   ├── rerank() or rerank_by_model()  # Rerank results                 │
│      │   └── Process into chunks with scores                                 │
│      │                                                                       │
│      ├── rerank()           # Lines 295-332 - Token + Vector reranking       │
│      │   └── Our TokenReranker.rerank()                                      │
│      │                                                                       │
│      └── rerank_by_model()  # Lines 334-355 - Cross-encoder reranking        │
│          └── Our CrossEncoderReranker.rerank()                               │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 RAGFlow's FulltextQueryer (query.py)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                  RAGFlow FulltextQueryer Breakdown                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  class FulltextQueryer:                                                      │
│      │                                                                       │
│      ├── __init__()                                                          │
│      │   ├── self.tw = term_weight.Dealer()  # TF-IDF weighting              │
│      │   ├── self.syn = synonym.Dealer()     # Synonym expansion             │
│      │   └── query_fields = [                # Field boosting                │
│      │       "title_tks^10",      # Title is 10x more important              │
│      │       "content_ltks^2",    # Content is 2x base                       │
│      │       "question_tks^20",   # Questions are 20x                        │
│      │       ...                                                             │
│      │   ]                                                                   │
│      │                                                                       │
│      ├── question()           # Lines 41-174 - Build BM25 query              │
│      │   ├── Normalize text (lowercase, remove special chars)                │
│      │   ├── Tokenize with weights                                           │
│      │   ├── Add synonyms for expansion                                      │
│      │   ├── Build complex ES query with boosts                              │
│      │   └── Return MatchTextExpr + keywords                                 │
│      │                                                                       │
│      ├── hybrid_similarity()  # Lines 176-184                                │
│      │   └── Our _cosine_similarity + token_similarity                       │
│      │                                                                       │
│      └── token_similarity()   # Lines 186-198                                │
│          └── Our _token_similarity with term weights                         │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.3 Key Insights from RAGFlow

1. **Field Boosting**: RAGFlow boosts different fields:
   ```python
   query_fields = [
       "title_tks^10",      # Title matches are 10x important
       "important_kwd^30",  # Important keywords 30x
       "question_tks^20",   # Question matches 20x
       "content_ltks^2",    # Regular content 2x
   ]
   ```

2. **Synonym Expansion**: RAGFlow expands queries with synonyms:
   ```python
   syn = self.syn.lookup(tk)  # "car" -> ["automobile", "vehicle"]
   ```

3. **Weighted Fusion**: Default 5% BM25 + 95% Vector:
   ```python
   fusionExpr = FusionExpr("weighted_sum", topk, {"weights": "0.05,0.95"})
   ```

4. **Multi-stage Reranking**:
   - Token similarity (TF-IDF weighted)
   - Vector similarity (cosine)
   - Cross-encoder (if available)
   - PageRank scores (for linked documents)
   - Tag features (categorical boosts)

---

## 5. Testing Your Implementation

### 5.1 Sample Documents

Create a test script:

```python
# tests/test_data.py
"""Sample documents for testing"""

SAMPLE_DOCUMENTS = [
    {
        "title": "Introduction to Machine Learning",
        "content": """
        Machine learning is a subset of artificial intelligence that enables 
        computers to learn and improve from experience without being explicitly 
        programmed. It focuses on developing algorithms that can access data 
        and use it to learn for themselves.
        
        There are three main types of machine learning:
        1. Supervised learning: The algorithm learns from labeled training data
        2. Unsupervised learning: The algorithm finds patterns in unlabeled data
        3. Reinforcement learning: The algorithm learns through trial and error
        
        Common applications include image recognition, natural language processing,
        recommendation systems, and autonomous vehicles.
        """
    },
    {
        "title": "Deep Learning Fundamentals",
        "content": """
        Deep learning is a subset of machine learning that uses neural networks
        with multiple layers (deep neural networks) to learn representations
        of data with multiple levels of abstraction.
        
        Key concepts in deep learning:
        - Neurons: Basic computational units
        - Layers: Groups of neurons
        - Weights: Parameters that are learned
        - Activation functions: Non-linear transformations
        - Backpropagation: Algorithm for training neural networks
        
        Popular deep learning frameworks include TensorFlow, PyTorch, and JAX.
        Applications include computer vision, speech recognition, and generative AI.
        """
    },
    {
        "title": "Natural Language Processing",
        "content": """
        Natural Language Processing (NLP) is a field of AI that focuses on
        the interaction between computers and human language. It enables
        machines to read, understand, and derive meaning from human languages.
        
        Key NLP tasks include:
        - Tokenization: Breaking text into words or subwords
        - Named Entity Recognition: Identifying proper nouns
        - Sentiment Analysis: Determining emotional tone
        - Machine Translation: Converting between languages
        - Question Answering: Responding to questions about text
        
        Modern NLP heavily relies on transformer models like BERT, GPT, and T5.
        """
    },
    {
        "title": "Vector Databases for AI",
        "content": """
        Vector databases are specialized databases designed to store and
        query high-dimensional vectors efficiently. They are essential for
        AI applications that rely on similarity search.
        
        Popular vector databases include:
        - Pinecone: Managed vector database service
        - Weaviate: Open-source vector search engine
        - Qdrant: Vector similarity search engine
        - Milvus: Open-source vector database
        - Elasticsearch: Also supports dense vectors
        
        Key operations:
        - Insert: Add vectors with metadata
        - Search: Find k-nearest neighbors
        - Filter: Combine vector search with metadata filters
        """
    },
    {
        "title": "RAG Systems Overview",
        "content": """
        Retrieval-Augmented Generation (RAG) combines information retrieval
        with text generation to produce more accurate and grounded responses.
        
        RAG architecture components:
        1. Document ingestion: Parse and chunk documents
        2. Embedding: Convert chunks to vectors
        3. Indexing: Store in vector database
        4. Retrieval: Find relevant chunks for query
        5. Generation: Use LLM to generate response with context
        
        Benefits of RAG:
        - Reduced hallucinations
        - Up-to-date information
        - Source attribution
        - Domain-specific knowledge
        """
    },
    {
        "title": "BM25 Algorithm Explained",
        "content": """
        BM25 (Best Matching 25) is a ranking function used by search engines
        to estimate the relevance of documents to a search query.
        
        The BM25 formula considers:
        - Term Frequency (TF): How often the term appears in the document
        - Inverse Document Frequency (IDF): How rare the term is across all docs
        - Document length: Normalizes for document size
        
        BM25 parameters:
        - k1: Controls term frequency saturation (default ~1.2)
        - b: Controls length normalization (default ~0.75)
        
        BM25 is effective for:
        - Exact keyword matching
        - Rare term boosting
        - Proper noun search
        """
    },
    {
        "title": "Embedding Models Comparison",
        "content": """
        Embedding models convert text into dense vector representations that
        capture semantic meaning. Different models have different trade-offs.
        
        Popular embedding models:
        
        1. OpenAI text-embedding-ada-002
           - Dimensions: 1536
           - Quality: Excellent
           - Cost: Paid API
        
        2. sentence-transformers/all-MiniLM-L6-v2
           - Dimensions: 384
           - Quality: Good
           - Cost: Free, runs locally
        
        3. BAAI/bge-small-en-v1.5
           - Dimensions: 384
           - Quality: Very good for RAG
           - Cost: Free, runs locally
        
        4. Cohere embed-v3
           - Dimensions: 1024
           - Quality: Excellent
           - Cost: Paid API
        """
    },
    {
        "title": "Hybrid Search Benefits",
        "content": """
        Hybrid search combines multiple search methods to improve retrieval
        quality. The most common combination is BM25 + vector search.
        
        Why use hybrid search?
        
        BM25 strengths:
        - Exact keyword matching
        - Works well with rare terms
        - No semantic drift
        - Interpretable results
        
        Vector search strengths:
        - Semantic understanding
        - Handles synonyms naturally
        - Works across languages
        - Finds conceptually similar content
        
        Hybrid combines both:
        - Score fusion (weighted sum)
        - Reciprocal rank fusion
        - Learn-to-rank combination
        
        RAGFlow uses 5% BM25 + 95% vector by default.
        """
    },
    {
        "title": "Reranking Strategies",
        "content": """
        Reranking is a second-stage retrieval process that re-scores
        initial results using a more sophisticated model.
        
        Why rerank?
        - Initial retrieval is fast but approximate
        - Reranking is slower but more accurate
        - Two-stage approach balances speed and quality
        
        Reranking approaches:
        
        1. Cross-encoders:
           - Score (query, document) pairs together
           - More accurate than bi-encoders
           - Examples: MS MARCO, BGE reranker
        
        2. Token overlap:
           - Simple term matching
           - Fast, no ML required
           - Used as fallback
        
        3. Learn-to-rank:
           - Train model on relevance labels
           - Combines multiple features
           - Most sophisticated
        """
    },
    {
        "title": "Chunking Strategies for RAG",
        "content": """
        Chunking is the process of splitting documents into smaller pieces
        for embedding and retrieval. The chunking strategy significantly
        impacts RAG quality.
        
        Common chunking strategies:
        
        1. Fixed-size chunking:
           - Split by character count
           - Simple but may break sentences
           - Add overlap for context
        
        2. Semantic chunking:
           - Split by paragraphs or sections
           - Preserves semantic units
           - Variable chunk sizes
        
        3. Recursive chunking:
           - Try multiple splitters
           - Fall back to smaller units
           - LangChain's default approach
        
        4. Document-type specific:
           - PDFs: By page or section
           - Code: By function or class
           - HTML: By tags
        
        Chunk size recommendations:
        - Small (256 tokens): More precise, more chunks
        - Medium (512 tokens): Balanced
        - Large (1024 tokens): More context, fewer chunks
        """
    }
]
```

### 5.2 Test Script

```python
# tests/test_search.py
"""
Test script for the Mini RAG system.

Run this after starting the services:
1. docker-compose up -d
2. uvicorn app.main:app --reload
3. python tests/test_search.py
"""
import requests
import time

BASE_URL = "http://localhost:8000"

# Test documents
from test_data import SAMPLE_DOCUMENTS


def test_ingest():
    """Test document ingestion"""
    print("\n" + "="*60)
    print("TESTING DOCUMENT INGESTION")
    print("="*60)
    
    doc_ids = []
    for i, doc in enumerate(SAMPLE_DOCUMENTS):
        response = requests.post(
            f"{BASE_URL}/api/documents/",
            json={
                "title": doc["title"],
                "content": doc["content"]
            }
        )
        
        if response.status_code == 200:
            result = response.json()
            doc_ids.append(result["doc_id"])
            print(f"✅ Ingested: {doc['title']}")
            print(f"   - Doc ID: {result['doc_id']}")
            print(f"   - Chunks: {result['chunks_created']}")
        else:
            print(f"❌ Failed: {doc['title']}")
            print(f"   - Error: {response.text}")
    
    return doc_ids


def test_search():
    """Test hybrid search"""
    print("\n" + "="*60)
    print("TESTING HYBRID SEARCH")
    print("="*60)
    
    queries = [
        # Exact keyword match (BM25 should help)
        "What is BM25?",
        
        # Semantic query (vector should help)
        "How do computers understand human language?",
        
        # Mixed query
        "What are the benefits of combining keyword and semantic search?",
        
        # Specific term (BM25 strength)
        "sentence-transformers embedding dimensions",
        
        # Conceptual query (vector strength)
        "How to make AI responses more accurate and grounded?",
    ]
    
    for query in queries:
        print(f"\n🔍 Query: '{query}'")
        
        response = requests.post(
            f"{BASE_URL}/api/search/",
            json={
                "question": query,
                "top_k": 3,
                "use_rerank": True
            }
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"   Found {result['total']} results:")
            
            for i, r in enumerate(result["results"][:3], 1):
                print(f"\n   [{i}] {r['title']}")
                print(f"       Score: {r['score']:.4f} "
                      f"(BM25: {r['bm25_score']:.4f}, "
                      f"Vector: {r['vector_score']:.4f})")
                print(f"       Content: {r['content'][:100]}...")
        else:
            print(f"   ❌ Error: {response.text}")


def test_stats():
    """Test index statistics"""
    print("\n" + "="*60)
    print("INDEX STATISTICS")
    print("="*60)
    
    response = requests.get(f"{BASE_URL}/api/documents/stats")
    if response.status_code == 200:
        stats = response.json()
        print(f"Total documents: {stats['total_docs']}")
        print(f"Index size: {stats['size_bytes'] / 1024:.2f} KB")


if __name__ == "__main__":
    print("🚀 Mini RAG Test Suite")
    print("Make sure services are running:")
    print("  - docker-compose up -d")
    print("  - uvicorn app.main:app --reload")
    
    # Wait for services
    time.sleep(2)
    
    # Run tests
    doc_ids = test_ingest()
    time.sleep(1)  # Wait for indexing
    
    test_stats()
    test_search()
    
    print("\n" + "="*60)
    print("✅ All tests completed!")
    print("="*60)
```

---

## 🎯 Checkpoints

Use this checklist to track your progress:

- [ ] **Setup**
  - [ ] Install dependencies (`uv sync` or `pip install -e .`)
  - [ ] Start Elasticsearch (`docker-compose up -d`)
  - [ ] Configure `.env` file

- [ ] **Index 10 documents into Elasticsearch**
  - [ ] Create index mapping with text + vector fields
  - [ ] Implement `ESIndexer.create_index()`
  - [ ] Implement `ESIndexer.index_chunk()`
  - [ ] Test: Documents appear in ES

- [ ] **Implement BM25 search**
  - [ ] Implement `Tokenizer.tokenize()`
  - [ ] Build match query for `content_ltks`
  - [ ] Test: Keyword searches return results

- [ ] **Add vector search with embeddings**
  - [ ] Load sentence-transformers model
  - [ ] Implement `Embedder.encode()`
  - [ ] Add KNN search to query
  - [ ] Test: Semantic queries work

- [ ] **Combine with weighted fusion**
  - [ ] Implement `HybridSearcher.search()`
  - [ ] Combine BM25 + vector scores
  - [ ] Test: Both search types contribute

- [ ] **Add reranking**
  - [ ] Implement `TokenReranker.rerank()`
  - [ ] Test: Results improve after reranking
  - [ ] (Optional) Add cross-encoder reranking

---

## 📚 Next Steps

After completing this phase, you'll understand:

1. **How BM25 works** - Term frequency, IDF, field boosting
2. **How vector search works** - Embeddings, cosine similarity
3. **Why hybrid search** - Complementary strengths
4. **How reranking improves results** - Two-stage retrieval

**Phase 2** will cover document parsing and chunking strategies - the "input" side of RAG.

**Phase 3** will cover LLM generation with citations - the "output" side of RAG.
