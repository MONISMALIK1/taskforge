"""
config.py — Application settings loaded from environment / .env file.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Ollama
    ollama_endpoint: str = "http://localhost:11434"
    ollama_model: str    = "llama3.2"
    ollama_timeout: int  = 90          # seconds per LLM call

    # Agent behaviour
    max_tool_calls: int  = 10          # prevent runaway loops
    task_timeout: int    = 120         # seconds before a running task is killed
    tool_top_k: int      = 5           # TF-IDF tool selection: send top-k relevant tools

    # Database
    database_url: str = "sqlite:///./taskforge.db"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
