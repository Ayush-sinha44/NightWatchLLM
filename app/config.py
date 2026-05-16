"""
Gateway configuration — loaded from environment variables.
Create a .env file (see .env.example) and never commit it.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Groq
    groq_api_key: str
    model: str = "llama-3.3-70b-versatile"   # fast free-tier Groq model
    default_max_tokens: int = 1024

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: list[str] = ["*"]


settings = Settings()