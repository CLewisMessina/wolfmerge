# app/database.py - Railway PostgreSQL URL Fix
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

# FIXED: Ensure proper async URL format for Railway
def get_async_database_url():
    """Convert Railway's database URL to proper async format"""
    db_url = settings.database_url
    
    # Railway typically provides postgresql:// URLs
    if db_url.startswith("postgresql://") and "asyncpg" not in db_url:
        async_url = db_url.replace("postgresql://", "postgresql+asyncpg://")
        logger.info(f"Converted database URL to async format")
        return async_url
    elif db_url.startswith("postgres://"):
        # Handle legacy postgres:// format
        async_url = db_url.replace("postgres://", "postgresql+asyncpg://")
        logger.info(f"Converted legacy postgres:// URL to async format")
        return async_url
    else:
        # Already in correct format or unknown format
        return db_url

# Create async engine with Railway-compatible URL
try:
    async_db_url = get_async_database_url()
    logger.info(f"Creating async engine with URL: {async_db_url[:30]}...")
    
    engine = create_async_engine(
        async_db_url,
        echo=settings.debug,
        pool_pre_ping=True,
        pool_recycle=300,
        pool_size=10,  # Reduced for Railway
        max_overflow=20  # Reduced for Railway
    )
    
    logger.info("✅ Database engine created successfully")
    
except Exception as e:
    logger.error(f"❌ Failed to create database engine: {e}")
    logger.error(f"Database URL format: {settings.database_url[:30]}...")
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
            # Create tables
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
            existing_workspace = await session.get(Workspace, "00000000-0000-0000-0000-000000000001")
            if existing_workspace:
                logger.info("Demo data already exists, skipping initialization")
                return
            
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
            
            demo_analyst = User(
                id=uuid.UUID("00000000-0000-0000-0000-000000000003"),
                email="analyst@beispiel-gmbh.de", 
                name="Thomas Müller",
                workspace_id=demo_workspace.id,
                role="analyst",
                language_preference="de",
                hashed_password="demo_password_hash",
                is_active=True,
                email_verified=True,
                gdpr_consent_date=datetime.now(timezone.utc),
                data_processing_consent=True
            )
            
            # Add to session
            session.add(demo_workspace)
            session.add(demo_admin)
            session.add(demo_analyst)
            
            # Create German compliance templates
            await _create_german_compliance_templates(session)
            
            await session.commit()
            
            logger.info(
                "✅ Demo data initialized successfully",
                workspace_id=str(demo_workspace.id),
                users_created=2,
                templates_created=True
            )
            
    except Exception as e:
        logger.error(f"❌ Failed to initialize demo data: {e}")
        raise

async def _create_german_compliance_templates(session: AsyncSession):
    """Create German industry-specific compliance templates"""
    
    templates = [
        {
            "name": "DSGVO Basis-Compliance für Automotive",
            "industry": "automotive",
            "framework": "gdpr",
            "german_authority": "BfDI",
            "legal_requirements": {
                "required_documents": [
                    "Datenschutzerklärung",
                    "Verfahrensverzeichnis nach Art. 30 DSGVO",
                    "Datenschutz-Folgenabschätzung für Fahrzeugdaten",
                    "Auftragsverarbeitungsverträge",
                    "Mitarbeiterschulung Datenschutz"
                ],
                "dsgvo_articles": ["Art. 6", "Art. 9", "Art. 13", "Art. 30", "Art. 32", "Art. 35"]
            },
            "checklist_items": [
                "Rechtsgrundlage für Fahrzeugdatenverarbeitung dokumentiert",
                "Informationspflichten gegenüber Fahrzeughaltern erfüllt",
                "Sicherheitsmaßnahmen für Connected Car Daten implementiert",
                "Löschkonzept für Telematikdaten vorhanden",
                "Drittlandsübermittlungen rechtlich abgesichert"
            ],
            "compliance_controls": {
                "technical_measures": [
                    "Verschlüsselung von Fahrzeugdaten",
                    "Zugriffskontrollen auf Telematik-Systeme",
                    "Protokollierung von Datenzugriffen"
                ],
                "organizational_measures": [
                    "Datenschutz-Managementsystem",
                    "Regelmäßige Mitarbeiterschulungen",
                    "Incident Response Prozess"
                ]
            }
        }
    ]
    
    for template_data in templates:
        template = ComplianceTemplate(
            name=template_data["name"],
            industry=template_data["industry"],
            framework=template_data["framework"],
            german_authority=template_data["german_authority"],
            legal_requirements=template_data["legal_requirements"],
            checklist_items=template_data["checklist_items"],
            compliance_controls=template_data.get("compliance_controls", {}),
            language="de",
            version="1.0"
        )
        
        session.add(template)

async def health_check() -> dict:
    """Check database connectivity and health"""
    try:
        async with AsyncSessionLocal() as session:
            # Test basic connectivity
            result = await session.execute(text("SELECT 1"))
            db_responsive = result.scalar() == 1
            
            # Test table existence
            tables_query = text("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
            """)
            tables_result = await session.execute(tables_query)
            tables = [row[0] for row in tables_result.fetchall()]
            
            # Count demo data
            workspace_count = 0
            user_count = 0
            try:
                workspace_result = await session.execute(text("SELECT COUNT(*) FROM workspaces"))
                workspace_count = workspace_result.scalar()
                
                user_result = await session.execute(text("SELECT COUNT(*) FROM users"))
                user_count = user_result.scalar()
            except Exception:
                pass  # Tables might not exist yet
            
            return {
                "database_connected": db_responsive,
                "tables_created": len(tables) > 0,
                "table_count": len(tables),
                "table_names": tables,
                "workspace_count": workspace_count,
                "user_count": user_count,
                "demo_data_loaded": workspace_count > 0 and user_count > 0,
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