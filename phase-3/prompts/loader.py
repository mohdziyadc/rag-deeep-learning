from pathlib import Path

PROMPT_DIR = Path(__file__).parent

def load_prompt(name:str) -> str:
    path = PROMPT_DIR / f"f{name}.md"
    return path.read_text(encoding="utf-8").strip()