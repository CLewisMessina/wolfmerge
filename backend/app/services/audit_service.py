# Comprehensive fix for the workspace foreign key violation issue
# This addresses both the immediate problem and makes the system more robust

# 1. Fix for audit_service.py - Add error handling for foreign key violations
# backend/app/services/audit_service.py

from typing import Optional, Dict, Any, List
from datetime import datetime, timezone, timedelta
import structlog
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, desc
from sqlalchemy.exc import IntegrityError
import uuid

from app.models.database import AuditLog, Workspace, User
from app.config import settings

logger = structlog.get_logger()

class AuditService:
    """GDPR-compliant audit logging service for German enterprises"""
    
    def __init__(self, db_session: AsyncSession):
        self.db = db_session
    
    def _get_utc_now(self) -> datetime:
        """Get current UTC time as timezone-naive datetime for database compatibility"""
        return datetime.now(timezone.utc).replace(tzinfo=None)
    
    def _calculate_retention_date(self, days: int) -> datetime:
        """Calculate retention date as timezone-naive datetime"""
        return self._get_utc_now() + timedelta(days=days)
    
    async def _workspace_exists(self, workspace_id: str) -> bool:
        """Check if workspace exists in database"""
        try:
            result = await self.db.execute(
                select(Workspace).where(Workspace.id == uuid.UUID(workspace_id))
            )
            return result.scalar_one_or_none() is not None
        except Exception:
            return False
    
    async def log_action(
        self,
        workspace_id: str,
        user_id: Optional[str],
        action: str,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        gdpr_basis: Optional[str] = None,
        data_category: Optional[str] = None
    ) -> Optional[str]:
        """
        Log user action for comprehensive audit trail
        Returns audit log ID for reference, or None if logging failed
        
        NEW: Added error handling for foreign key violations
        """
        
        # Check if workspace exists before attempting to log
        if not await self._workspace_exists(workspace_id):
            logger.warning(
                "Audit log skipped - workspace not found",
                workspace_id=workspace_id,
                action=action,
                message="Consider creating workspace or using valid workspace_id"
            )
            return None
        
        # Calculate retention date (timezone-naive for database)
        retain_until = self._calculate_retention_date(settings.audit_retention_days)
        
        # Determine GDPR basis if not provided
        if not gdpr_basis:
            gdpr_basis = self._determine_gdpr_basis(action, resource_type)
        
        # Determine data category if not provided
        if not data_category:
            data_category = self._categorize_data_processing(action, resource_type)
        
        try:
            # Create audit entry with timezone-naive datetimes
            audit_entry = AuditLog(
                workspace_id=uuid.UUID(workspace_id),
                user_id=uuid.UUID(user_id) if user_id else None,
                action=action,
                resource_type=resource_type,
                resource_id=uuid.UUID(resource_id) if resource_id else None,
                details=details or {},
                ip_address=ip_address,
                user_agent=user_agent,
                gdpr_basis=gdpr_basis,
                data_category=data_category,
                status="success",
                retain_until=retain_until
            )
            
            self.db.add(audit_entry)
            await self.db.commit()
            
            logger.info(
                "Audit entry created successfully",
                audit_id=str(audit_entry.id),
                workspace_id=workspace_id,
                action=action,
                resource_type=resource_type,
                gdpr_basis=gdpr_basis
            )
            
            return str(audit_entry.id)
            
        except IntegrityError as e:
            await self.db.rollback()
            logger.warning(
                "Failed to write audit log - integrity constraint violation",
                workspace_id=workspace_id,
                action=action,
                error=str(e),
                message="Continuing without audit log to prevent service disruption"
            )
            return None
            
        except Exception as e:
            await self.db.rollback()
            logger.error(
                "Unexpected error writing audit log",
                workspace_id=workspace_id,
                action=action,
                error=str(e),
                exc_info=True
            )
            return None
    
    async def log_error(
        self,
        workspace_id: str,
        user_id: Optional[str],
        action: str,
        error_message: str,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """Log failed action for audit trail with error handling"""
        
        # Check if workspace exists
        if not await self._workspace_exists(workspace_id):
            logger.warning(
                "Error audit log skipped - workspace not found",
                workspace_id=workspace_id,
                action=action,
                error_message=error_message
            )
            return None
        
        try:
            retain_until = self._calculate_retention_date(settings.audit_retention_days)
            
            audit_entry = AuditLog(
                workspace_id=uuid.UUID(workspace_id),
                user_id=uuid.UUID(user_id) if user_id else None,
                action=action,
                resource_type=resource_type,
                resource_id=uuid.UUID(resource_id) if resource_id else None,
                details=details or {},
                gdpr_basis=self._determine_gdpr_basis(action, resource_type),
                data_category=self._categorize_data_processing(action, resource_type),
                status="failure",
                error_message=error_message,
                retain_until=retain_until
            )
            
            self.db.add(audit_entry)
            await self.db.commit()
            
            logger.error(
                "Audit error logged successfully",
                audit_id=str(audit_entry.id),
                workspace_id=workspace_id,
                action=action,
                error=error_message
            )
            
            return str(audit_entry.id)
            
        except IntegrityError as e:
            await self.db.rollback()
            logger.warning(
                "Failed to write error audit log - integrity constraint violation",
                workspace_id=workspace_id,
                action=action,
                original_error=error_message,
                audit_error=str(e)
            )
            return None
            
        except Exception as e:
            await self.db.rollback()
            logger.error(
                "Unexpected error writing error audit log",
                workspace_id=workspace_id,
                action=action,
                original_error=error_message,
                audit_error=str(e),
                exc_info=True
            )
            return None
    
    def _determine_gdpr_basis(self, action: str, resource_type: Optional[str]) -> str:
        """Determine GDPR legal basis for processing"""
        
        # Map actions to GDPR Article 6 legal bases
        legal_basis_mapping = {
            # Legitimate interest (Art. 6(1)(f))
            "document_analysis": "Art. 6(1)(f) - Legitimate interest for service operation",
            "compliance_check": "Art. 6(1)(f) - Legitimate interest for regulatory compliance",
            "audit_log_creation": "Art. 6(1)(f) - Legitimate interest for security and audit",
            
            # Contractual necessity (Art. 6(1)(b))
            "user_login": "Art. 6(1)(b) - Contract performance for service provision",
            "workspace_access": "Art. 6(1)(b) - Contract performance for service access",
            
            # Legal obligation (Art. 6(1)(c))
            "audit_trail_maintenance": "Art. 6(1)(c) - Legal obligation for audit requirements",
            "data_retention_enforcement": "Art. 6(1)(c) - Legal obligation for data protection",
        }
        
        # Default to legitimate interest for service operation
        return legal_basis_mapping.get(action, "Art. 6(1)(f) - Legitimate interest for service operation")
    
    def _categorize_data_processing(self, action: str, resource_type: Optional[str]) -> str:
        """Categorize type of data being processed"""
        
        if "document" in action or resource_type == "document":
            return "compliance_analysis_data"
        elif "user" in action or resource_type == "user":
            return "user_management_data"
        elif "audit" in action:
            return "audit_trail_data"
        elif "workspace" in action or resource_type == "workspace":
            return "workspace_management_data"
        else:
            return "system_operation_data"


# 2. Enhanced init_demo_data function with better error handling
# backend/app/database.py (replacement for init_demo_data function)

async def init_demo_data():
    """Initialize demo workspace and data for Day 2 testing with enhanced error handling"""
    
    demo_workspace_id = uuid.UUID("00000000-0000-0000-0000-000000000001")
    demo_admin_id = uuid.UUID("00000000-0000-0000-0000-000000000002")
    
    try:
        async with AsyncSessionLocal() as session:
            logger.info("Starting demo data initialization...")
            
            # Check if demo workspace already exists
            try:
                existing_workspace = await session.get(Workspace, demo_workspace_id)
                if existing_workspace:
                    logger.info(
                        "Demo data already exists",
                        workspace_id=str(demo_workspace_id),
                        workspace_name=existing_workspace.name
                    )
                    return
            except Exception as e:
                logger.warning(f"Error checking existing workspace: {e}")
                # Continue with creation attempt
            
            # Create demo workspace for German SME
            try:
                demo_workspace = Workspace(
                    id=demo_workspace_id,
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
                
                session.add(demo_workspace)
                await session.flush()  # Get the ID assigned
                
                logger.info(
                    "Demo workspace created",
                    workspace_id=str(demo_workspace.id),
                    workspace_name=demo_workspace.name
                )
                
            except Exception as e:
                logger.error(f"Failed to create demo workspace: {e}")
                await session.rollback()
                raise
            
            # Create demo admin user
            try:
                demo_admin = User(
                    id=demo_admin_id,
                    email="admin@beispiel-gmbh.de",
                    name="Dr. Maria Schmidt",
                    workspace_id=demo_workspace.id,
                    role="admin",
                    german_certification="TÜV Certified DPO",
                    language_preference="de",
                    hashed_password="demo_password_hash",
                    is_active=True,
                    email_verified=True,
                    # Use timezone-naive datetime for GDPR consent
                    gdpr_consent_date=datetime.now(timezone.utc).replace(tzinfo=None),
                    data_processing_consent=True
                )
                
                session.add(demo_admin)
                await session.flush()
                
                logger.info(
                    "Demo admin user created",
                    user_id=str(demo_admin.id),
                    user_email=demo_admin.email
                )
                
            except Exception as e:
                logger.error(f"Failed to create demo admin user: {e}")
                await session.rollback()
                raise
            
            # Commit all changes
            await session.commit()
            
            logger.info(
                "✅ Demo data initialized successfully",
                workspace_id=str(demo_workspace_id),
                admin_user_id=str(demo_admin_id),
                workspace_name="Beispiel GmbH Compliance Team"
            )
            
    except Exception as e:
        logger.error(
            "❌ Failed to initialize demo data",
            error=str(e),
            workspace_id=str(demo_workspace_id),
            exc_info=True
        )
        # Re-raise the exception so startup fails if demo data is critical
        # Comment out the raise below if you want startup to continue without demo data
        raise  # This will cause startup to fail, which might be what we want for debugging


# 3. Workspace validation helper
# backend/app/services/workspace_service.py (new file)

from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import structlog
import uuid

from app.models.database import Workspace, User

logger = structlog.get_logger()

class WorkspaceService:
    """Service for workspace operations and validation"""
    
    def __init__(self, db_session: AsyncSession):
        self.db = db_session
    
    async def ensure_workspace_exists(self, workspace_id: str) -> bool:
        """
        Ensure workspace exists, create default if needed
        Returns True if workspace exists or was created successfully
        """
        try:
            # Check if workspace exists
            result = await self.db.execute(
                select(Workspace).where(Workspace.id == uuid.UUID(workspace_id))
            )
            existing_workspace = result.scalar_one_or_none()
            
            if existing_workspace:
                logger.debug(f"Workspace {workspace_id} exists")
                return True
            
            # If it's the demo workspace ID, create it
            if workspace_id == "00000000-0000-0000-0000-000000000001":
                return await self._create_demo_workspace()
            
            logger.warning(f"Workspace {workspace_id} does not exist and cannot be auto-created")
            return False
            
        except Exception as e:
            logger.error(f"Error checking workspace existence: {e}")
            return False
    
    async def _create_demo_workspace(self) -> bool:
        """Create demo workspace on-demand"""
        try:
            demo_workspace = Workspace(
                id=uuid.UUID("00000000-0000-0000-0000-000000000001"),
                name="Auto-Created Demo Workspace",
                organization="Demo Organization",
                country="DE",
                industry="technology",
                compliance_frameworks=["gdpr"],
                german_authority="BfDI",
                language_preference="de",
                timezone="Europe/Berlin",
                gdpr_consent=True,
                audit_level="standard",
                subscription_tier="sme",
                max_documents=100,
                max_users=10
            )
            
            self.db.add(demo_workspace)
            await self.db.commit()
            
            logger.info("Demo workspace created successfully on-demand")
            return True
            
        except Exception as e:
            await self.db.rollback()
            logger.error(f"Failed to create demo workspace on-demand: {e}")
            return False


# 4. Update enhanced_compliance_analyzer.py to handle audit failures gracefully
# Add this to the analyze_documents_for_workspace method:

# Replace the audit service call with:
try:
    await self.audit_service.log_action(
        workspace_id=workspace_id,
        user_id=user_id,
        action="document_analysis_started",
        resource_type="analysis",
        details={
            "framework": framework.value,
            "document_count": len(files),
            "total_size": sum(size for _, _, size in files)
        }
    )
except Exception as audit_error:
    # Don't fail the entire analysis if audit logging fails
    logger.warning(
        "Audit logging failed but continuing with analysis",
        workspace_id=workspace_id,
        audit_error=str(audit_error)
    )

# And at the end of successful analysis:
try:
    await self.audit_service.log_action(
        workspace_id=workspace_id,
        user_id=user_id,
        action="document_analysis_completed",
        resource_type="analysis",
        resource_id=str(analysis_record.id),
        details={
            "framework": framework.value,
            "documents_analyzed": len(files),
            "total_chunks": len(all_chunks_metadata),
            "processing_time": processing_time,
            "compliance_score": compliance_report.compliance_score
        }
    )
except Exception as audit_error:
    logger.warning(
        "Completion audit logging failed",
        workspace_id=workspace_id,
        analysis_id=str(analysis_record.id),
        audit_error=str(audit_error)
    )


# 5. Debug endpoint to check workspace status
# Add to compliance_router.py:

@router.get("/debug/workspace/{workspace_id}")
async def debug_workspace_status(
    workspace_id: str,
    db_session: AsyncSession = Depends(get_db_session)
):
    """Debug endpoint to check workspace status"""
    try:
        from app.services.workspace_service import WorkspaceService
        
        workspace_service = WorkspaceService(db_session)
        workspace_exists = await workspace_service.ensure_workspace_exists(workspace_id)
        
        # Get workspace details if it exists
        result = await db_session.execute(
            select(Workspace).where(Workspace.id == uuid.UUID(workspace_id))
        )
        workspace = result.scalar_one_or_none()
        
        return {
            "workspace_id": workspace_id,
            "exists": workspace_exists,
            "workspace": {
                "name": workspace.name if workspace else None,
                "organization": workspace.organization if workspace else None,
                "created_at": workspace.created_at.isoformat() if workspace else None
            } if workspace else None,
            "demo_workspace_id": "00000000-0000-0000-0000-000000000001",
            "is_demo_workspace": workspace_id == "00000000-0000-0000-0000-000000000001"
        }
        
    except Exception as e:
        return {
            "workspace_id": workspace_id,
            "error": str(e),
            "exists": False
        }