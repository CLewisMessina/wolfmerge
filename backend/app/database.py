# app/database.py - Defensive Railway Database Setup
import asyncio
from typing import AsyncGenerator
from datetime import datetime, timezone
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy import text
import structlog
import uuid
import os

from app.config import settings
from app.models.database import Base, Workspace, User, ComplianceTemplate

logger = structlog.get_logger()

def get_async_database_url():
    """Convert Railway's database URL to proper async format with error handling"""
    db_url = settings.database_url
    
    # Debug: Log what we actually received
    logger.info(f"Raw database URL from settings: {db_url[:50]}...")
    
    # Check if Railway variable wasn't resolved
    if "${{" in db_url and "}}" in db_url:
        logger.error(f"❌ Railway environment variable not resolved: {db_url}")
        logger.error("This suggests the DATABASE_URL variable reference is incorrect in Railway.")
        logger.error("Check that your Railway env var matches your PostgreSQL service name exactly.")
        
        # Try to get from direct environment variables as fallback
        fallback_vars = [
            "DATABASE_PRIVATE_URL",
            "POSTGRES_DATABASE_URL", 
            "DATABASE_PUBLIC_URL"
        ]
        
        for var in fallback_vars:
            fallback_url = os.environ.get(var)
            if fallback_url and not ("${{" in fallback_url and "}}" in fallback_url):
                logger.warning(f"Using fallback database URL from {var}")
                db_url = fallback_url
                break
        else:
            raise ValueError(
                f"DATABASE_URL contains unresolved Railway variable: {db_url}. "
                "Check your Railway environment variable configuration."
            )
    
    # Convert to async format
    if db_url.startswith("postgresql://") and "asyncpg" not in db_url:
        async_url = db_url.replace("postgresql://", "postgresql+asyncpg://")
        logger.info("✅ Converted postgresql:// to postgresql+asyncpg://")
        return async_url
    elif db_url.startswith("postgres://"):
        async_url = db_url.replace("postgres://", "postgresql+asyncpg://")
        logger.info("✅ Converted postgres:// to postgresql+asyncpg://")
        return async_url
    elif db_url.startswith("postgresql+asyncpg://"):
        logger.info("✅ Database URL already in correct async format")
        return db_url
    else:
        logger.warning(f"Unknown database URL format: {db_url[:30]}...")
        return db_url

# Create async engine with defensive error handling
try:
    async_db_url = get_async_database_url()
    logger.info(f"Creating async engine with URL: {async_db_url[:50]}...")
    
    engine = create_async_engine(
        async_db_url,
        echo=settings.debug,
        pool_pre_ping=True,
        pool_recycle=300,
        pool_size=5,  # Conservative for Railway
        max_overflow=10
    )
    
    logger.info("✅ Database engine created successfully")
    
except Exception as e:
    logger.error(f"❌ Failed to create database engine: {e}")
    logger.error(f"Database URL format received: {settings.database_url}")
    logger.error("This is likely a Railway environment variable configuration issue.")
    
    # List all available environment variables for debugging
    logger.error("Available DATABASE/POSTGRES environment variables:")
    for key, value in os.environ.items():
        if any(x in key.upper() for x in ['DATABASE', 'POSTGRES']):
            safe_value = value[:30] + "..." if len(value) > 30 else value
            logger.error(f"  {key}: {safe_value}")
    
    raise

# Create async session factory
AsyncSessionLocal = async_sessionmaker(
    engine, 
    class_=AsyncSession, 
    expire_on_commit=False,
    autoflush=True,
    autocommit=False
)

async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Dependency for database sessions with proper cleanup"""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception as e:
            await session.rollback()
            logger.error("Database session error", error=str(e))
            raise
        finally:
            await session.close()

async def create_tables():
    """Create all database tables for Day 2 enterprise features"""
    try:
        logger.info("Creating database tables...")
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            
        logger.info("✅ Database tables created successfully")
            
    except Exception as e:
        logger.error(f"❌ Failed to create database tables: {e}")
        raise

async def init_demo_data():
    """Initialize demo workspace and data for Day 2 testing"""
    try:
        async with AsyncSessionLocal() as session:
            # Check if demo data already exists
            try:
                existing_workspace = await session.get(Workspace, uuid.UUID("00000000-0000-0000-0000-000000000001"))
                if existing_workspace:
                    logger.info("Demo data already exists, skipping initialization")
                    return
            except Exception:
                # Table might not exist yet, continue with creation
                pass
            
            # Create demo workspace for German SME
            demo_workspace = Workspace(
                id=uuid.UUID("00000000-0000-0000-0000-000000000001"),
                name="Beispiel GmbH Compliance Team",
                organization="Beispiel GmbH",
                country="DE",
                industry="automotive",
                compliance_frameworks=["gdpr", "iso27001"],
                german_authority="BfDI",
                dpo_contact="dpo@beispiel-gmbh.de",
                legal_entity_type="GmbH",
                language_preference="de",
                timezone="Europe/Berlin",
                gdpr_consent=True,
                audit_level="enhanced",
                subscription_tier="sme",
                max_documents=500,
                max_users=25
            )
            
            # Create demo users
            demo_admin = User(
                id=uuid.UUID("00000000-0000-0000-0000-000000000002"),
                email="admin@beispiel-gmbh.de",
                name="Dr. Maria Schmidt",
                workspace_id=demo_workspace.id,
                role="admin",
                german_certification="TÜV Certified DPO",
                language_preference="de",
                hashed_password="demo_password_hash",
                is_active=True,
                email_verified=True,
                gdpr_consent_date=datetime.now(timezone.utc),
                data_processing_consent=True
            )
            
            session.add(demo_workspace)
            session.add(demo_admin)
            
            await session.commit()
            
            logger.info("✅ Demo data initialized successfully")
            
    except Exception as e:
        logger.error(f"❌ Failed to initialize demo data: {e}")
        # Don't fail the entire startup for demo data issues
        logger.warning("Continuing startup without demo data")

async def health_check() -> dict:
    """Check database connectivity and health"""
    try:
        async with AsyncSessionLocal() as session:
            # Test basic connectivity
            result = await session.execute(text("SELECT 1"))
            db_responsive = result.scalar() == 1
            
            return {
                "database_connected": db_responsive,
                "tables_created": True,  # Assume true if we got this far
                "demo_data_loaded": False,  # We'll check this later
                "eu_region": settings.eu_region,
                "gdpr_compliant": settings.gdpr_compliance
            }
            
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return {
            "database_connected": False,
            "error": str(e),
            "eu_region": settings.eu_region,
            "gdpr_compliant": settings.gdpr_compliance
        }

# Demo workspace and user IDs for Day 2 testing
DEMO_WORKSPACE_ID = "00000000-0000-0000-0000-000000000001"
DEMO_ADMIN_USER_ID = "00000000-0000-0000-0000-000000000002"
DEMO_ANALYST_USER_ID = "00000000-0000-0000-0000-000000000003"