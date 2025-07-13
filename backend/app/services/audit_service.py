# Fix for audit_service.py datetime timezone issues
# The database expects timezone-naive datetimes but we're providing timezone-aware ones

# In backend/app/services/audit_service.py
# Update the datetime handling throughout the file:

from typing import Optional, Dict, Any, List
from datetime import datetime, timezone, timedelta
import structlog
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, desc
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
    ) -> str:
        """
        Log user action for comprehensive audit trail
        Returns audit log ID for reference
        """
        
        # Calculate retention date (timezone-naive for database)
        retain_until = self._calculate_retention_date(settings.audit_retention_days)
        
        # Determine GDPR basis if not provided
        if not gdpr_basis:
            gdpr_basis = self._determine_gdpr_basis(action, resource_type)
        
        # Determine data category if not provided
        if not data_category:
            data_category = self._categorize_data_processing(action, resource_type)
        
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
            # created_at will be auto-set by the model's default
        )
        
        self.db.add(audit_entry)
        await self.db.commit()
        
        logger.info(
            "Audit entry created",
            audit_id=str(audit_entry.id),
            workspace_id=workspace_id,
            action=action,
            resource_type=resource_type,
            gdpr_basis=gdpr_basis
        )
        
        return str(audit_entry.id)
    
    async def log_error(
        self,
        workspace_id: str,
        user_id: Optional[str],
        action: str,
        error_message: str,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> str:
        """Log failed action for audit trail"""
        
        # Use timezone-naive datetime for database
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
            # created_at will be auto-set by the model's default
        )
        
        self.db.add(audit_entry)
        await self.db.commit()
        
        logger.error(
            "Audit error logged",
            audit_id=str(audit_entry.id),
            workspace_id=workspace_id,
            action=action,
            error=error_message
        )
        
        return str(audit_entry.id)
    
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
            entries_returned=len(audit_entries),
            filters_applied={
                "action": action_filter,
                "user": user_filter,
                "date_from": date_from,
                "date_to": date_to
            }
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
    
    async def get_compliance_audit_report(
        self,
        workspace_id: str,
        report_type: str = "monthly"
    ) -> Dict[str, Any]:
        """Generate compliance audit report for German authorities"""
        
        # Calculate date range based on report type (timezone-naive)
        end_date = self._get_utc_now()
        if report_type == "weekly":
            start_date = end_date - timedelta(days=7)
        elif report_type == "monthly":
            start_date = end_date - timedelta(days=30)
        elif report_type == "quarterly":
            start_date = end_date - timedelta(days=90)
        else:
            start_date = end_date - timedelta(days=365)  # yearly
        
        # Get audit entries for period
        audit_entries = await self.get_workspace_audit_trail(
            workspace_id=workspace_id,
            limit=10000,  # High limit for comprehensive report
            date_from=start_date,
            date_to=end_date
        )
        
        # Analyze audit data
        report = {
            "report_metadata": {
                "workspace_id": workspace_id,
                "report_type": report_type,
                "period_start": start_date.isoformat(),
                "period_end": end_date.isoformat(),
                "generated_at": end_date.isoformat(),
                "total_entries": len(audit_entries),
                "gdpr_compliant": True,
                "retention_policy": f"{settings.audit_retention_days} days"
            },
            "activity_summary": self._analyze_activity_patterns(audit_entries),
            "gdpr_compliance": self._analyze_gdpr_compliance(audit_entries),
            "security_events": self._identify_security_events(audit_entries),
            "user_activity": self._analyze_user_activity(audit_entries),
            "data_processing_summary": self._summarize_data_processing(audit_entries),
            "recommendations": self._generate_audit_recommendations(audit_entries)
        }
        
        logger.info(
            "Compliance audit report generated",
            workspace_id=workspace_id,
            report_type=report_type,
            entries_analyzed=len(audit_entries),
            security_events=len(report["security_events"])
        )
        
        return report
    
    def _analyze_gdpr_compliance(self, audit_entries: List[AuditLog]) -> Dict[str, Any]:
        """Analyze GDPR compliance aspects of audit trail"""
        
        gdpr_analysis = {
            "legal_basis_coverage": {},
            "data_category_processing": {},
            "retention_compliance": True,
            "data_subject_rights_requests": 0,
            "consent_management_events": 0,
            "data_deletion_events": 0,
            "breach_indicators": []
        }
        
        # Use timezone-naive datetime for comparison
        current_time = self._get_utc_now()
        
        for entry in audit_entries:
            # Legal basis analysis
            basis = entry.gdpr_basis or "Unknown"
            gdpr_analysis["legal_basis_coverage"][basis] = gdpr_analysis["legal_basis_coverage"].get(basis, 0) + 1
            
            # Data category analysis
            category = entry.data_category or "Unknown"
            gdpr_analysis["data_category_processing"][category] = gdpr_analysis["data_category_processing"].get(category, 0) + 1
            
            # Check retention compliance
            if entry.retain_until and entry.retain_until < current_time:
                gdpr_analysis["retention_compliance"] = False
            
            # Count specific GDPR-related actions
            if "data_subject_request" in entry.action:
                gdpr_analysis["data_subject_rights_requests"] += 1
            elif "consent" in entry.action:
                gdpr_analysis["consent_management_events"] += 1
            elif "delete" in entry.action or "purge" in entry.action:
                gdpr_analysis["data_deletion_events"] += 1
            
            # Identify potential breach indicators
            if entry.status == "failure" and any(
                keyword in entry.action for keyword in ["unauthorized", "breach", "violation"]
            ):
                gdpr_analysis["breach_indicators"].append({
                    "timestamp": entry.created_at.isoformat() if entry.created_at else None,
                    "action": entry.action,
                    "error": entry.error_message
                })
        
        return gdpr_analysis
    
    # Rest of the methods remain the same, just ensure any datetime operations use timezone-naive datetimes
    
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