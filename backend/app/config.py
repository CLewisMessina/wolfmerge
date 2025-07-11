# app/config.py - Railway-Optimized with Authority Engine Support - FIXED VERSION
import os
from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import Field
import structlog

# Initialize logger for this module
logger = structlog.get_logger()

class Settings(BaseSettings):
    # OpenAI Configuration
    openai_api_key: str
    openai_model: str = "gpt-4o-mini"
    
    # Server Configuration
    api_title: str = "WolfMerge Enterprise Compliance Platform"
    api_version: str = "2.0.0"
    debug: bool = False
    environment: str = "production"
    
    # Database Configuration - SIMPLIFIED for Railway
    database_url: str = Field(alias="DATABASE_URL")
    
    # EU Cloud Configuration
    eu_region: bool = True
    gdpr_compliance: bool = True
    data_residency: str = "EU"
    audit_logging: bool = True
    
    # Team Workspace Configuration
    enable_workspaces: bool = True
    max_workspace_size_mb: int = 100
    max_users_per_workspace: int = 50
    workspace_retention_days: int = 365
    
    # Docling Configuration
    docling_enabled: bool = True
    chunk_size: int = 512
    chunk_overlap: int = 50
    max_chunks_per_document: int = 100
    enable_table_extraction: bool = True
    enable_image_extraction: bool = False
    
    # Security Configuration
    secret_key: str
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7
    
    # File Processing - ENHANCED for Authority Engine Support
    max_file_size_mb: int = 50
    max_files_per_batch: int = 10
    
    # FIXED: Missing properties that caused the API errors
    max_total_file_size_mb: int = 200  # Total batch size limit in MB
    max_total_file_size: int = 200 * 1024 * 1024  # Total batch size limit in bytes
    
    allowed_extensions: List[str] = [".txt", ".md", ".pdf", ".docx", ".doc"]
    
    # GDPR Compliance
    data_retention_hours: int = 24
    audit_retention_days: int = 90
    consent_required: bool = True
    auto_delete_processed_content: bool = True
    
    # Performance & Caching
    enable_redis_cache: bool = True
    cache_ttl_seconds: int = 3600
    enable_async_processing: bool = True
    
    # German Compliance Specific
    german_legal_frameworks: List[str] = ["gdpr", "dsgvo", "bdsg"]
    default_language: str = "de"
    supported_languages: List[str] = ["de", "en"]
    
    # Authority Engine Configuration - NEW
    big4_authority_engine_enabled: bool = True
    authority_detection_threshold: float = 0.5  # Minimum confidence for authority detection
    german_authority_priority_boost: float = 0.3  # Priority boost for German content
    
    # CORS Configuration
    cors_origins: List[str] = [
        "http://localhost:3000",
        "https://app.wolfmerge.com",
        "https://dev-app.wolfmerge.com"
    ]
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"  # Ignore extra Railway environment variables

    @property
    def database_url_async(self) -> str:
        """Convert database URL to async format when needed - handled in database.py"""
        return self.database_url
    
    @property
    def database_url_sync(self) -> str:
        """Convert async database URL to sync for migrations"""
        if "asyncpg" in self.database_url:
            return self.database_url.replace("postgresql+asyncpg://", "postgresql://")
        return self.database_url
    
    @property
    def max_file_size_bytes(self) -> int:
        """Convert individual file size MB to bytes"""
        return self.max_file_size_mb * 1024 * 1024
    
    @property
    def max_workspace_size_bytes(self) -> int:
        """Convert workspace size MB to bytes"""
        return self.max_workspace_size_mb * 1024 * 1024
    
    @property
    def max_total_file_size_bytes(self) -> int:
        """Convert total batch size MB to bytes - FIXED: Uses the correct property"""
        return self.max_total_file_size_mb * 1024 * 1024
    
    @property
    def is_production(self) -> bool:
        """Check if running in production"""
        return self.environment.lower() == "production"
    
    @property
    def is_development(self) -> bool:
        """Check if running in development"""
        return self.environment.lower() in ["development", "dev", "local"]

# Safe instantiation with Railway debugging - IMPROVED: Better logging practices
try:
    settings = Settings()
    
    # FIXED: Improved logging to avoid exposing sensitive information
    logger.info("Settings loaded successfully", 
                environment=settings.environment,
                database_connection="configured",
                openai_configured=bool(settings.openai_api_key),
                secret_key_configured=bool(settings.secret_key))
    
    logger.info("Authority Engine configuration",
                big4_enabled=settings.big4_authority_engine_enabled,
                detection_threshold=settings.authority_detection_threshold)
    
    logger.info("File processing limits",
                individual_file_mb=settings.max_file_size_mb,
                batch_total_mb=settings.max_total_file_size_mb,
                workspace_mb=settings.max_workspace_size_mb,
                max_files_per_batch=settings.max_files_per_batch)
    
except Exception as e:
    logger.error("Settings loading failed", error=str(e))
    
    # Debug Railway environment variables - IMPROVED: More secure logging
    logger.info("Checking environment configuration...")
    
    relevant_vars = ['DATABASE_URL', 'OPENAI_API_KEY', 'SECRET_KEY', 'ENVIRONMENT', 'DEBUG']
    for var in relevant_vars:
        value = os.environ.get(var)
        if value:
            if 'KEY' in var or 'SECRET' in var:
                logger.debug(f"{var}: configured ({len(value)} characters)")
            else:
                logger.debug(f"{var}: {value}")
        else:
            logger.warning(f"{var}: NOT SET")
    
    # Log Railway-specific environment variables (without sensitive data)
    railway_vars = [key for key in os.environ.keys() 
                   if any(prefix in key.upper() for prefix in ['RAILWAY', 'DATABASE', 'API']) 
                   and 'KEY' not in key and 'SECRET' not in key]
    
    if railway_vars:
        logger.info("Railway environment variables detected", 
                   variable_count=len(railway_vars),
                   variables=railway_vars[:5])  # Only show first 5 for brevity
    
    raise

# Export settings instance
__all__ = ["settings"]