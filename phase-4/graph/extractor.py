import re
from prompts import GRAPH_EXTRACTION_PROMPT
from llm.client import LLMClient
from models.schemas import GraphEntity, GraphRelation, GraphExtractionResult

DEFAULT_TUPLE_DELIMITER = "<|>"
DEFAULT_RECORD_DELIMITER= "##"
DEFAULT_COMPLETION_DELIMITER="<|COMPLETE|>"

class GraphExtractor:
    
    def __init__(self, llm: LLMClient, entity_types: list[str]) -> None:
        self.llm = llm
        self.entity_types = entity_types
    

    def extract(self, text: str) -> GraphExtractionResult:
        """
        Extract entities + relations from a chunk. Mirrors
        graphrag/general/graph_extractor.py parsing flow.
        """

        system_prompt = GRAPH_EXTRACTION_PROMPT.format(
            entity_types=", ".join(self.entity_types),
            tuple_delimiter=DEFAULT_TUPLE_DELIMITER,
            record_delimiter=DEFAULT_RECORD_DELIMITER,
            completion_delimiter=DEFAULT_COMPLETION_DELIMITER
        )

        raw = self.llm.chat(system_prompt=system_prompt, user_prompt=text)
        records = raw.split(DEFAULT_RECORD_DELIMITER)

        entities: list[GraphEntity] = []
        relations: list[GraphRelation] = []

        for record in records:
            match = re.match(r"\((.*)\)", record)

            if not match:
                continue
            
            parts = match.group(1).split(DEFAULT_TUPLE_DELIMITER)
            """
            group(1) returns the text captured by the first parenthesized group in your regex.
            So for re.search(r"\((.*)\)", record):
            - group(0) is the whole match (including the parentheses)
            - group(1) is just the part inside the parentheses
            Example:
            record = '("entity"<|>"Alex")'
            match = re.search(r"\((.*)\)", record)
            match.group(0)  # -> '("entity"<|>"Alex")'
            match.group(1)  # -> '"entity"<|>"Alex"'
            """
            if not parts:
                continue

            part_type = parts[0].strip().strip('"')

            if part_type == "entity" and len(parts) >= 4:
                entities.append(GraphEntity(
                    entity_name=parts[1].strip('" '),
                    entity_type=parts[2].strip('" '),
                    description=parts[3].strip('" ')
                ))

            if part_type == "relationship" and len(parts) >= 5:
                relations.append(GraphRelation(
                    src_id=parts[1].strip('" '),
                    tgt_id=parts[2].strip('" '),
                    description=parts[3].strip('" '),
                    strength=float(parts[4])
                ))

            """
            .strip('" ') removes both surrounding quotes and stray spaces. 
            .strip('"') would only remove quotes, leaving leading/trailing spaces intact.
            Example:
            s = ' "Alex" '
            s.strip('"')   # -> ' "Alex" '
            s.strip('" ')  # -> 'Alex'
            """

        return GraphExtractionResult(entities=entities, relations=relations)
