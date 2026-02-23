import re
import numpy as np
from embedder.embedder import embedder


class CitationInserter:

    def __init__(self, max_citations_per_sentence: int = 4) -> None:
        self.max_citations = max_citations_per_sentence

    def _split_sentences(self, text: str) -> list[str]:
        """
        this function is going to split the input into 
        a list of strs by splitting it by the punctuation
        signifying the end of a sentence.
        After that it merges by the split punctuations
        to form a an array of sentences
        """
        parts = re.split(r"([。？！.!?]\s)", text)
        if len(parts) <= 1:
            return [text]
        
        merged = []
        
        for i in range(0, len(parts) - 1, 2):
            merged.append(parts[i] + parts[i+1])

        if len(parts) % 2 != 0:
            merged.append(parts[-1])
        return [sentence for sentence in merged if sentence.strip()]
    

    def insert_citations(self, answer: str, chunk_content: list[str]) -> tuple[str, set[int]]:
        """
        Add citations by matching answer sentences to retrieved chunks
        via hybrid similarity, mirroring RAGFlow's insert_citations.
        """
        if not chunk_content:
            return answer, set()

        sentences = self._split_sentences(answer)
        embedder.load()
        sentence_vecs, _ = embedder.encode(sentences)
        chunk_vecs, _ = embedder.encode(chunk_content)

        citations = {}

        for i, svec in enumerate(sentence_vecs):
            # cosine similarity
            # a small constant 1e-8 is added to the denominator
            # to prevent divided by 0 errors
            # each sentence vector is compared against
            # the chunk_content vectors
            similarity_array = np.dot(chunk_vecs, svec) / (
                np.linalg.norm(chunk_vecs, axis=1) * np.linalg.norm(svec) + 1e-8
            )

            top_ids = list(np.argsort(similarity_array))[::-1][: self.max_citations]
            citations[i] = [int(idx) for idx in top_ids if similarity_array[idx] > 0.3]
        
        used = set()
        out = []
        for i, sentence in enumerate(sentences):
            out.append(sentence)
            if i in citations and citations[i]:
                for idx in citations[i]:
                    out.append(f"[ID:{idx}]")
                    used.add(idx)

        """
        ragflow doesnt just use semantic cosine similarity
        it uses via a hybrid similarity, vector similarity
        + token similarity (for keyword similarity)
        In RAGFlow's insert_citations, the hybrid similarity weights
        are set in the function signature:
        - tkweight = 0.1 (token overlap)
        - vtweight = 0.9 (vector/embedding)
        Is it like BM25 + vector?  
        Conceptually yes (lexical + semantic mix), but it's not BM25 here. 
        It uses a tokenizer-based overlap similarity, not full BM25 scoring. 
        BM25 + vector fusion is used earlier in retrieval; the citation insertion step 
        uses this lighter hybrid similarity for matching sentences to chunks.
        """
        
        return "".join(out), used


"""
Actual explanation of this function.
---
Example inputs
answer  
"RAG adds citations after generation. Cosine similarity is used to match sentences."
chunks (list of strings)
[
  "RAG adds citations after generation using similarity.",
  "Cosine similarity compares embedding vectors.",
  "Token budgets prevent context overflow."
]
Assume max_citations = 2.
---
Line-by-line breakdown + intermediate outputs
1) Empty-chunks guard
if not chunks:
    return answer, set()
- If there are no chunks, nothing to cite.  
- Not triggered in this example.
---
2) Split into sentences
sentences = self._split_sentences(answer)
Output (sentences):
[
  "RAG adds citations after generation. ",
  "Cosine similarity is used to match sentences."
]
(Each sentence keeps punctuation so we can place citations at sentence ends.)
---
3) Load embedding model
embedder.load()
- Ensures the embedding model is ready.
---
4) Encode sentences and chunks
sent_vecs, _ = embedder.encode(sentences)
chunk_vecs, _ = embedder.encode(chunks)
Output shapes:
- sent_vecs: one vector per sentence  
  → shape: (2, dim)  
- chunk_vecs: one vector per chunk  
  → shape: (3, dim)
(Actual numbers depend on the embedding model.)
---
5) Compute similarities per sentence
citations = {}
for i, svec in enumerate(sent_vecs):
    sims = np.dot(chunk_vecs, svec) / (
        np.linalg.norm(chunk_vecs, axis=1) * np.linalg.norm(svec) + 1e-8
    )
For sentence 0 (“RAG adds citations…”) suppose cosine sims are:
sims = [0.82, 0.20, 0.05]
For sentence 1 (“Cosine similarity…”) suppose:
sims = [0.22, 0.78, 0.09]
---
6) Pick top chunk IDs
top_ids = list(np.argsort(sims)[::-1])[: self.max_citations]
For sentence 0:
- argsort returns the indices by which we can 
sort the array in ascending order of similarities
- np.argsort([0.82, 0.20, 0.05]) → [2, 1, 0]
- [::-1] → [0, 1, 2]
- [:2] → [0, 1]
For sentence 1:
- np.argsort([0.22, 0.78, 0.09]) → [2, 0, 1]
- [::-1] → [1, 0, 2]
- [:2] → [1, 0]
---
7) Apply similarity threshold
citations[i] = [int(idx) for idx in top_ids if sims[idx] > 0.3]
Threshold is 0.3.
For sentence 0:
- top_ids [0, 1] → sims [0.82, 0.20]
- Only 0.82 > 0.3
- citations[0] = [0]
For sentence 1:
- top_ids [1, 0] → sims [0.78, 0.22]
- Only 0.78 > 0.3
- citations[1] = [1]
Intermediate citations dict:
{
  0: [0],
  1: [1]
}
---
8) Build output + track used IDs
used = set()
out = []
for i, sentence in enumerate(sentences):
    out.append(sentence)
    if i in citations and citations[i]:
        for idx in citations[i]:
            out.append(f" [ID:{idx}]")
            used.add(idx)
Processing sentence 0:
- Output becomes:  
  "RAG adds citations after generation.  [ID:0]"
Processing sentence 1:
- Output becomes:  
  "RAG adds citations after generation.  [ID:0]Cosine similarity is used to match sentences. [ID:1]"
used becomes:
{0, 1}
---
9) Final return
return "".join(out), used
Final answer:
"RAG adds citations after generation.  [ID:0]Cosine similarity is used to match sentences. [ID:1]"
Final used IDs:
{0, 1}
---
Summary
- Input: raw answer + list of chunk texts  
- Process: sentence split → embeddings → cosine similarity → top chunk IDs → append citations  
- Output: answer with [ID:x] citations + set of used chunk indices

"""