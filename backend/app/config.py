# app/config.py
import os
from typing import List
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # OpenAI Configuration
    openai_api_key: str
    openai_model: str = "gpt-4o-mini"
    
    # Server Configuration
    api_title: str = "WolfMerge Compliance API"
    api_version: str = "1.0.0"
    debug: bool = False
    
    # CORS Configuration
    cors_origins: List[str] = ["http://localhost:3000"]
    
    # File Processing
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    max_files: int = 5
    allowed_extensions: List[str] = [".txt", ".md"]
    
    # Future: EU Cloud Configuration (Day 2)
    eu_region: bool = True
    gdpr_compliance: bool = True
    
    class Config:
        env_file = ".env"

settings = Settings()