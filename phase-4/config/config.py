
from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    es_host: str
    es_index: str

    class Config:
        env_file = Path(__file__).resolve().parents[1] / ".env"


settings = Settings()


"""
- / ".env" → path join, not string concatenation. In pathlib, / is overloaded to mean “join path segments”.
Example:
# Suppose __file__ = /Users/me/project/app/config.py
Path(__file__).resolve()        # /Users/me/project/app/config.py
Path(__file__).resolve().parent # /Users/me/project/app
.parents[1]                     # /Users/me/project
/ ".env"                        # /Users/me/project/.env
"""
