

from generation.citation_inserter import CitationInserter
from generation.context_packer import ContextPacker
from generation.llm_client import LLMClient
from generation.prompt_builder import PromptBuilder
from models.dtos import ChatMessage, ChatResponse, ReferenceBundle, ReferenceChunk, RetrievedChunk, ChatRes


class Generator:
    def __init__(self, llm: LLMClient, model_name:str, max_tokens:int) -> None:
        self.prompts = PromptBuilder(model_name, max_tokens)
        self.packer = ContextPacker(model_name=model_name, max_tokens=max_tokens)
        self.llm = llm
        self.citer = CitationInserter()
    

    def _build_reference(self, chunks: list[RetrievedChunk], used_ids: set[int]) -> ReferenceBundle:
        """
        this method attaches the og chunk for the cited reference
        """
        refs = []
        for i, chunk in enumerate(chunks):
            if used_ids and i not in used_ids:
                continue
            refs.append(
                ReferenceChunk(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    content=chunk.content,
                    metadata=chunk.metadata
                )
            )
        return ReferenceBundle(chunks=refs, doc_aggs=[])

    def generate(
        self, 
        question: str, 
        messages: list[ChatMessage], 
        chunks: list[RetrievedChunk],
        qoute: bool
    ) -> ChatResponse:
        """
            End-to-end generation: build prompt, fit context, call LLM,
            then insert citations and return references.
        """
        system_prompt = self.prompts.build_system_prompt(chunks=chunks, qoute=qoute)
        packed = self.packer.fit_messages(system_prompt, messages + [ChatMessage(role="user", content=question)])

        answer = self.llm.chat(packed)
        used = set()
        if qoute and chunks and "[ID:" not in answer:
            answer, used = self.citer.insert_citations(answer=answer, chunk_content=[c.content for c in chunks])
        
        references = self._build_reference(chunks, used)
        # returning system_prompt for fun 😋
        return ChatResponse(answer=answer, reference=references, prompt=system_prompt)

    def stream(
        self,
        messages: list[ChatMessage],
        chunks: list[RetrievedChunk],
        quote: bool
    ):
        """
        Stream generation while preserving the same prompt and
        citation logic as non-streaming responses.
        """
        system_prompt = self.prompts.build_system_prompt(chunks=chunks, qoute=qoute)
        packed = self.packer.fit_messages(system_prompt, messages + [ChatMessage(role="user", content=question)])

        full_answer = ""
        for delta in self.llm.stream_chat(messages):
            full_answer += delta
            yield {"type": "delta", "data": delta}
        
        used = set()
        if quote and chunks and "[ID:" not in full_answer:
            full_answer, used = self.citer.insert_citations(
                answer=full_answer,
                chunk_content=[c.content for c in chunks]
            )
        
        reference = self._build_reference(chunks, used)

        """
        In the streaming path you're manually building SSE events and JSON-encoding them, 
        so the payload must be plain JSON-serializable data. 
        ChatResponse(...).model_dump() converts the Pydantic model into a 
        dict that json.dumps can serialize.
        In the non-streaming generate() path, you return the Pydantic model instance directly.
        FastAPI will serialize it automatically in the response, so you don't need model_dump() there.
        """

        yield {
            "type": "final",
            "data": ChatResponse(answer=full_answer, reference=reference, prompt=system_prompt).model_dump()
        }






    