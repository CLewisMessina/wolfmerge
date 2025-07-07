# app/config.py - Railway-Optimized for Your Environment
import os
from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import Field

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
    
    # File Processing
    max_file_size_mb: int = 50
    max_files_per_batch: int = 10
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
        """Convert MB to bytes"""
        return self.max_file_size_mb * 1024 * 1024
    
    @property
    def is_production(self) -> bool:
        """Check if running in production"""
        return self.environment.lower() == "production"

# Safe instantiation with Railway debugging
try:
    settings = Settings()
    print(f"✅ Settings loaded successfully for environment: {settings.environment}")
    print(f"✅ Database URL format: {settings.database_url[:20]}...")
    print(f"✅ OpenAI API key loaded: {settings.openai_api_key[:10]}...")
    print(f"✅ Secret key loaded: {len(settings.secret_key)} characters")
except Exception as e:
    print(f"❌ Settings loading failed: {e}")
    
    # Debug Railway environment variables
    print("\n🔍 Railway Environment Debug:")
    relevant_vars = ['DATABASE_URL', 'OPENAI_API_KEY', 'SECRET_KEY', 'ENVIRONMENT', 'DEBUG']
    for var in relevant_vars:
        value = os.environ.get(var)
        if value:
            if 'KEY' in var:
                print(f"  {var}: {value[:10]}... ({len(value)} chars)")
            else:
                print(f"  {var}: {value}")
        else:
            print(f"  {var}: ❌ NOT SET")
    
    print(f"\n📋 All environment variables starting with relevant prefixes:")
    for key in sorted(os.environ.keys()):
        if any(prefix in key.upper() for prefix in ['DATABASE', 'OPENAI', 'SECRET', 'API', 'DEBUG']):
            value = os.environ[key]
            if 'KEY' in key or 'SECRET' in key:
                print(f"  {key}: {value[:10]}... ({len(value)} chars)")
            else:
                print(f"  {key}: {value}")
    
    raise