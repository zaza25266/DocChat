from pydantic_settings import BaseSettings
from .constants import MAX_FILE_SIZE, MAX_TOTAL_SIZE, ALLOWED_TYPES

class Settings(BaseSettings):
    OPENAI_API_KEY: str = ""

    WATSONX_APIKEY: str = ""
    WATSONX_PROJECT_ID: str = "skills-network"
    WATSONX_URL: str = "https://us-south.ml.cloud.ibm.com"

    MAX_FILE_SIZE: int = MAX_FILE_SIZE
    MAX_TOTAL_SIZE: int = MAX_TOTAL_SIZE
    ALLOWED_TYPES: list = ALLOWED_TYPES

    CHROMA_DB_PATH: str = "./chroma_db"
    CHROMA_COLLECTION_NAME: str = "documents"

    VECTOR_SEARCH_K: int = 10
    HYBRID_RETRIEVER_WEIGHTS: list = [0.4, 0.6]

    LOG_LEVEL: str = "INFO"

    CACHE_DIR: str = "document_cache"
    CACHE_EXPIRE_DAYS: int = 7

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

settings = Settings()