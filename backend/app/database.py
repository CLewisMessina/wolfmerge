# app/database.py - Railway-compatible database setup
import asyncio
from typing import AsyncGenerator
from datetime import datetime, timezone
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy import text
import structlog
import uuid

from app.config import settings
from app.models.database import Base, Workspace, User, ComplianceTemplate

logger = structlog.get_logger()

# FIXED: Use the async URL property from settings
engine = create_async_engine(
    settings.database_url_async,  # Use the property that handles URL conversion
    echo=settings.debug,
    pool_pre_ping=True,
    pool_recycle=300,
    pool_size=20,
    max_overflow=30
)

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
        async with engine.begin() as conn:
            # Create tables
            await conn.run_sync(Base.metadata.create_all)
            
            logger.info("Database tables created successfully")
            
    except Exception as e:
        logger.error("Failed to create database tables", error=str(e))
        raise

# Rest of your database.py remains the same...