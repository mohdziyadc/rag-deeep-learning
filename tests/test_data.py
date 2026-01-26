

SAMPLE_DOCS = [
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