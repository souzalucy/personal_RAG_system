from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    groq_api_key: str = "your-groq-api-key-here"
    groq_model: str = "llama-3.3-70b-versatile"

    # Qdrant Cloud — set QDRANT_URL and QDRANT_API_KEY to connect
    qdrant_url: str = ""                 # Qdrant Cloud cluster URL (e.g. https://xxxxx.cloud.qdrant.io:6333)
    qdrant_api_key: str = ""             # Qdrant Cloud API key
    qdrant_collection: str = "documents"

    # Lua scripts directory
    lua_dir: str = "./lua"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
