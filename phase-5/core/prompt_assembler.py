from typing import Optional
from core.enums import MemoryType


class PromptAssembler:
    SYSTEM_PROMPT = """
**Memory Extraction Specialist**
You are an expert at analyzing conversations to extract structured memory.

{type_specific_instructions}

**OUTPUT REQUIREMENTS:**
1. Output MUST be valid JSON
2. Follow the specified output format exactly
3. Each extracted item MUST have: content, valid_at, invalid_at
4. Timestamps in ISO 8601 format
5. Only extract memory types specified above
6. Maximum {max_items} items per type
"""

    TYPE_INSTRUCTIONS = {
        "semantic": """
**EXTRACT SEMANTIC KNOWLEDGE:**
- Facts, definitions, stable relationships.
""",
        "episodic": """
**EXTRACT EPISODIC KNOWLEDGE:**
- Time-bound events and experiences.
""",
        "procedural": """
**EXTRACT PROCEDURAL KNOWLEDGE:**
- Steps, methods, and actionable processes.
""",
    }

    OUTPUT_TEMPLATES = {
        "semantic": '"semantic": [{"content": "...", "valid_at": "...", "invalid_at": ""}]',
        "episodic": '"episodic": [{"content": "...", "valid_at": "...", "invalid_at": ""}]',
        "procedural": '"procedural": [{"content": "...", "valid_at": "...", "invalid_at": ""}]',
    }

    USER_PROMPT = """
**CONVERSATION:**
{conversation}

**CONVERSATION TIME:** {conversation_time}
**CURRENT TIME:** {current_time}
"""

    @staticmethod
    def _valid_types(memory_types: list[str]) -> list[str]:
        all_types = {m.name.lower() for m in MemoryType}
        return [
            type for type in memory_types if type in all_types and type.lower() != "raw"
        ]

    @classmethod
    def assemble_system_prompt(cls, config: dict) -> str:
        types_to_extract = cls._valid_types(config["memory_type"])
        instructions = "\n".join(cls.TYPE_INSTRUCTIONS[t] for t in types_to_extract)
        output_format = ",\n".join(cls.OUTPUT_TEMPLATES[t] for t in types_to_extract)

        prompt = cls.SYSTEM_PROMPT.format(
            type_specific_instructions=instructions,
            max_items=config.get("max_items_per_type", 5),
        )

        prompt += f"""
        **REQUIRED OUTPUT FORMAT (JSON)**:
        ```json
        {{{output_format}}}
        ```
        """

        return prompt

    @classmethod
    def assemble_user_prompt(
        cls, conversation: str, convo_time: Optional[str], current_time: Optional[str]
    ) -> str:
        return cls.USER_PROMPT.format(
            conversation=conversation,
            conversation_time=convo_time or "Not Specified",
            current_time=current_time or "Not Specified",
        )
