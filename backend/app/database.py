# app/database.py - Day 2 EU Cloud Database Setup
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

# Create async engine for EU PostgreSQL database
engine = create_async_engine(
    settings.database_url,
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

async def init_demo_data():
    """Initialize demo workspace and data for Day 2 testing"""
    try:
        async with AsyncSessionLocal() as session:
            # Check if demo data already exists
            existing_workspace = await session.get(Workspace, "demo-workspace-eu")
            if existing_workspace:
                logger.info("Demo data already exists, skipping initialization")
                return
            
            # Create demo workspace for German SME
            demo_workspace = Workspace(
                id=uuid.UUID("00000000-0000-0000-0000-000000000001"),  # Fixed UUID for demo
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
                id=uuid.UUID("00000000-0000-0000-0000-000000000002"),  # Fixed UUID for demo
                email="admin@beispiel-gmbh.de",
                name="Dr. Maria Schmidt",
                workspace_id=demo_workspace.id,
                role="admin",
                german_certification="TÜV Certified DPO",
                language_preference="de",
                hashed_password="demo_password_hash",  # In production, use proper hashing
                is_active=True,
                email_verified=True,
                gdpr_consent_date=datetime.now(timezone.utc),
                data_processing_consent=True
            )
            
            demo_analyst = User(
                id=uuid.UUID("00000000-0000-0000-0000-000000000003"),  # Fixed UUID for demo
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
                "Demo data initialized successfully",
                workspace_id=str(demo_workspace.id),
                users_created=2,
                templates_created=True
            )
            
    except Exception as e:
        logger.error("Failed to initialize demo data", error=str(e))
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
        },
        {
            "name": "DSGVO Compliance für Healthcare",
            "industry": "healthcare",
            "framework": "gdpr",
            "german_authority": "BfDI",
            "legal_requirements": {
                "required_documents": [
                    "Patientendatenschutz-Richtlinie",
                    "Einwilligungserklärungen",
                    "Schweigepflichtvereinbarungen",
                    "DSFA für Gesundheitsdatenverarbeitung",
                    "Notfallzugriff-Regelungen"
                ],
                "dsgvo_articles": ["Art. 6", "Art. 9", "Art. 17", "Art. 20", "Art. 32", "Art. 35"]
            },
            "checklist_items": [
                "Besondere Kategorien von Gesundheitsdaten geschützt",
                "Patientenrechte (Auskunft, Löschung, Übertragbarkeit) gewährleistet",
                "Medizinische Schweigepflicht technisch umgesetzt",
                "Aufbewahrungsfristen gem. Berufsrecht eingehalten",
                "Notfallzugriff auf Patientendaten geregelt"
            ]
        },
        {
            "name": "DSGVO Compliance für Manufacturing",
            "industry": "manufacturing",
            "framework": "gdpr",
            "german_authority": "BfDI",
            "legal_requirements": {
                "required_documents": [
                    "Mitarbeiterdatenschutz-Richtlinie",
                    "Kundendatenverarbeitung-Verfahren",
                    "Lieferanten-Datenschutzvereinbarungen",
                    "IoT/Industrie 4.0 Datenschutzkonzept",
                    "Betriebsrat-Datenschutzvereinbarung"
                ],
                "dsgvo_articles": ["Art. 6", "Art. 13", "Art. 30", "Art. 32", "Art. 44-49"]
            },
            "checklist_items": [
                "Mitarbeiterdatenverarbeitung rechtskonform",
                "Kunden- und Lieferantendaten geschützt",
                "IoT-Sensordaten datenschutzkonform verarbeitet",
                "Internationale Lieferketten DSGVO-konform",
                "Betriebsrat bei Überwachungsmaßnahmen beteiligt"
            ]
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
        logger.error("Database health check failed", error=str(e))
        return {
            "database_connected": False,
            "error": str(e),
            "eu_region": settings.eu_region,
            "gdpr_compliant": settings.gdpr_compliance
        }

async def run_migrations():
    """Run database migrations (simplified for Day 2)"""
    try:
        # In production, this would use Alembic
        # For Day 2, we'll just recreate tables
        await create_tables()
        logger.info("Database migrations completed")
        
    except Exception as e:
        logger.error("Database migrations failed", error=str(e))
        raise

# Utility functions for Day 2 enterprise features

async def get_workspace_by_id(workspace_id: str) -> Workspace:
    """Get workspace by ID with error handling"""
    async with AsyncSessionLocal() as session:
        workspace = await session.get(Workspace, uuid.UUID(workspace_id))
        if not workspace:
            raise ValueError(f"Workspace {workspace_id} not found")
        return workspace

async def get_user_by_id(user_id: str) -> User:
    """Get user by ID with error handling"""
    async with AsyncSessionLocal() as session:
        user = await session.get(User, uuid.UUID(user_id))
        if not user:
            raise ValueError(f"User {user_id} not found")
        return user

# Demo workspace and user IDs for Day 2 testing
DEMO_WORKSPACE_ID = "00000000-0000-0000-0000-000000000001"
DEMO_ADMIN_USER_ID = "00000000-0000-0000-0000-000000000002"
DEMO_ANALYST_USER_ID = "00000000-0000-0000-0000-000000000003"