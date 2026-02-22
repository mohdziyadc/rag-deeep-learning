import tiktoken

from models.dtos import ChatMessage


class ContextPacker:

    def __init__(self, model_name: str, max_tokens:int) -> None:
        self.encoding = tiktoken.encoding_for_model(model_name=model_name)
        self.max_tokens = max_tokens

    def _count_tokens(self, text:str) -> int:
        return len(self.encoding.encode(text))
    

    def fit_messages(self, system_prompt:str, messages: list[ChatMessage]) -> list[dict]:
        """
        Trim chat history to fit a token budget while preserving the
        system prompt and the most recent turns, similar to RAGFlow's
        message_fit_in behavior.
        """
        
        packed = [{"role": "system", "content": system_prompt}]
        token_limit = self.max_tokens - self._count_tokens(system_prompt)

        # Keep latest messages first, then reverse back.
        reversed_msgs = list(reversed(messages))
        kept_after_trimming = []

        for msg in reversed_msgs:
            tokens = self._count_tokens(msg.content)

            if tokens > token_limit:
                continue
            kept_after_trimming.append(msg)
            token_limit -= tokens
        
        packed.extend({"role": m.role, "content": m.content} for m in reversed(kept_after_trimming))
        return packed

