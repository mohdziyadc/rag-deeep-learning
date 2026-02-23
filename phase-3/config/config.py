from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    llm_api_key: str
    llm_base_url: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-4o-mini"
    max_tokens: int = 4096

    class Config:
        env_file = ".env"


def get_settings() -> Settings:
    return Settings()