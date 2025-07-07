# app/config.py - Day 2 Enhanced
import os
from typing import List, Optional
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # OpenAI Configuration (from Day 1)
    openai_api_key: str
    openai_model: str = "gpt-4o-mini"
    
    # Server Configuration (enhanced)
    api_title: str = "WolfMerge Enterprise Compliance Platform"
    api_version: str = "2.0.0"
    debug: bool = False
    environment: str = "production"
    
    # Database Configuration (Day 2: EU Cloud)
    database_url: str
    database_url_async: Optional[str] = None
    redis_url: Optional[str] = None
    
    # EU Cloud Configuration
    eu_region: bool = True
    gdpr_compliance: bool = True
    data_residency: str = "EU"
    audit_logging: bool = True
    
    # Team Workspace Configuration (Day 2)
    enable_workspaces: bool = True
    max_workspace_size_mb: int = 100
    max_users_per_workspace: int = 50
    workspace_retention_days: int = 365
    
    # Docling Configuration (Day 2)
    docling_enabled: bool = True
    chunk_size: int = 512
    chunk_overlap: int = 50
    max_chunks_per_document: int = 100
    enable_table_extraction: bool = True
    enable_image_extraction: bool = False  # Day 3 feature
    
    # Security Configuration (Day 2)
    secret_key: str
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7
    
    # File Processing (Enhanced from Day 1)
    max_file_size_mb: int = 50  # Increased for enterprise documents
    max_files_per_batch: int = 10  # Increased for team workflows
    allowed_extensions: List[str] = [".txt", ".md", ".pdf", ".docx", ".doc"]
    
    # GDPR Compliance (Day 2)
    data_retention_hours: int = 24
    audit_retention_days: int = 90
    consent_required: bool = True
    auto_delete_processed_content: bool = True
    
    # Performance & Caching (Day 2)
    enable_redis_cache: bool = True
    cache_ttl_seconds: int = 3600
    enable_async_processing: bool = True
    
    # German Compliance Specific
    german_legal_frameworks: List[str] = ["gdpr", "dsgvo", "bdsg"]
    default_language: str = "de"
    supported_languages: List[str] = ["de", "en"]
    
    # CORS Configuration (enhanced)
    cors_origins: List[str] = [
        "http://localhost:3000",
        "https://app.wolfmerge.com",
        "https://dev-app.wolfmerge.com"
    ]
    
    class Config:
        env_file = ".env"
        case_sensitive = False

    @property
    def database_url_sync(self) -> str:
        """Convert async database URL to sync for migrations"""
        if self.database_url.startswith("postgresql+asyncpg://"):
            return self.database_url.replace("postgresql+asyncpg://", "postgresql://")
        return self.database_url
    
    @property
    def max_file_size_bytes(self) -> int:
        """Convert MB to bytes"""
        return self.max_file_size_mb * 1024 * 1024
    
    @property
    def is_production(self) -> bool:
        """Check if running in production"""
        return self.environment.lower() == "production"

# Global settings instance
settings = Settings()