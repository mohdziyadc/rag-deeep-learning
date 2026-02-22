import re
from jinja2 import Template
import tiktoken

from models.dtos import RetrievedChunk
from prompts.loader import load_prompt


class PromptBuilder:
    def __init__(
        self,
        model_name: str,
        max_tokens: int,
        knowledge_budget_ratio: float = 0.7
    ) -> None:
        self.system_template = Template(load_prompt("system"))
        self.citation_prompt = load_prompt("citation_prompt")
        self.encoding = tiktoken.encoding_for_model(model_name)
        # the max amount of tokens for {knowledge} section in the 
        # prompt
        self.knowledge_budget = int(max_tokens * knowledge_budget_ratio)

    def _count_tokens(self, text: str) -> int:
        return len(self.encoding.encode(text))
    
    def draw_node(self, key:str, value: str | None) -> str | None:
        if value is None:
            return None
        text = re.sub(r"\n+", " ", str(value)).strip()
        if not text:
            return None
        return f"├── {key}: {text}"

    def format_knowledge(self, chunks: list[RetrievedChunk]) -> str:
        """
        Build a tree-style knowledge block like RAGFlow's kb_prompt:
        - Trim chunks by a token budget
        - Provide stable [ID:i] anchors for citations
        - Include title/URL/metadata when available
        """
        lines = ["<context>"]
        used_tokens = 0
        for idx, chunk in enumerate(chunks):
            content = chunk.content or ""
            chunk_tokens = self._count_tokens(content)
            
            if used_tokens + chunk_tokens > self.knowledge_budget:
                break
            used_tokens += chunk_tokens


            lines.append(f"ID: {idx}")
            node = self.draw_node("Title", chunk.title)
            if node:
                lines.append(node)
            
            url = chunk.metadata.get("url") 
            node = self.draw_node("URL", url)

            if node:
                lines.append(node)

            for k,v in chunk.metadata.items():
                if k in {"url", "title"}:
                    continue
                node = self.draw_node(k,v)
                if node:
                    lines.append(node)
            lines.append(f"└── Content: {chunk.content}")
        lines.append("</context>")
        return "\n".join(lines)

    def build_system_prompt(self, chunks: list[RetrievedChunk], qoute: bool) -> str:
        knowledge = self.format_knowledge(chunks=chunks)
        base = self.system_template.render(knowledge=knowledge)

        if qoute and chunks:
            return base + "\n\n" + self.citation_prompt
        return 
        

"""
Output of format_knowledge (example):
<context>
ID: 0
├── Title: Phase 3 Guide
├── URL: https://example.com/phase-3
├── source: internal-docs
└── Content: Phase 3 builds a generation layer that injects retrieved chunks into a system prompt and adds citations post-generation.
ID: 1
├── Title: RAGFlow kb_prompt
└── Content: RAGFlow builds a tree-style context block with chunk IDs and metadata to stabilize citations and reduce hallucinations.
</context>
"""
