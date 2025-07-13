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
                resource_type=resource_type
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
                error=str(e)
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
                audit_error=str(e)
            )
            return None

    async def get_workspace_audit_trail(
        self,
        workspace_id: str,
        limit: int = 100,
        action_filter: Optional[str] = None,
        user_filter: Optional[str] = None,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None
    ) -> List[AuditLog]:
        """Get filtered audit trail for workspace"""
        
        query = select(AuditLog).where(
            AuditLog.workspace_id == uuid.UUID(workspace_id)
        )
        
        # Apply filters (convert timezone-aware dates to naive if needed)
        if action_filter:
            query = query.where(AuditLog.action.ilike(f"%{action_filter}%"))
        
        if user_filter:
            query = query.where(AuditLog.user_id == uuid.UUID(user_filter))
        
        if date_from:
            # Ensure timezone-naive for database comparison
            if date_from.tzinfo is not None:
                date_from = date_from.replace(tzinfo=None)
            query = query.where(AuditLog.created_at >= date_from)
        
        if date_to:
            # Ensure timezone-naive for database comparison
            if date_to.tzinfo is not None:
                date_to = date_to.replace(tzinfo=None)
            query = query.where(AuditLog.created_at <= date_to)
        
        # Order by most recent first and limit results
        query = query.order_by(desc(AuditLog.created_at)).limit(limit)
        
        result = await self.db.execute(query)
        audit_entries = result.scalars().all()
        
        logger.info(
            "Audit trail retrieved",
            workspace_id=workspace_id,
            entries_returned=len(audit_entries)
        )
        
        return audit_entries

    async def cleanup_expired_audit_logs(self) -> int:
        """Clean up audit logs that have exceeded retention period"""
        
        # Use timezone-naive datetime for database comparison
        current_time = self._get_utc_now()
        
        # Find expired entries
        query = select(AuditLog).where(
            AuditLog.retain_until < current_time
        )
        
        result = await self.db.execute(query)
        expired_entries = result.scalars().all()
        
        # Delete expired entries
        deleted_count = 0
        for entry in expired_entries:
            await self.db.delete(entry)
            deleted_count += 1
        
        if deleted_count > 0:
            await self.db.commit()
            
            logger.info(
                "Expired audit logs cleaned up",
                deleted_count=deleted_count,
                cleanup_date=current_time.isoformat()
            )
        
        return deleted_count

    def _determine_gdpr_basis(self, action: str, resource_type: Optional[str]) -> str:
        """Determine GDPR legal basis for processing"""
        
        # Map actions to GDPR Article 6 legal bases
        legal_basis_mapping = {
            # Legitimate interest (Art. 6(1)(f))
            "document_analysis": "Art. 6(1)(f) - Legitimate interest for service operation",
            "document_analysis_started": "Art. 6(1)(f) - Legitimate interest for service operation", 
            "document_analysis_completed": "Art. 6(1)(f) - Legitimate interest for service operation",
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