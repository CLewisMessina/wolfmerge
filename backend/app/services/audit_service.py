# app/services/audit_service.py - Day 2 GDPR Compliance Audit Service
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
        
        # Calculate retention date based on GDPR requirements
        retain_until = datetime.now(timezone.utc) + timedelta(days=settings.audit_retention_days)
        
        # Determine GDPR basis if not provided
        if not gdpr_basis:
            gdpr_basis = self._determine_gdpr_basis(action, resource_type)
        
        # Determine data category if not provided
        if not data_category:
            data_category = self._categorize_data_processing(action, resource_type)
        
        # Create audit entry
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
        
        retain_until = datetime.now(timezone.utc) + timedelta(days=settings.audit_retention_days)
        
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
        
        # Apply filters
        if action_filter:
            query = query.where(AuditLog.action.ilike(f"%{action_filter}%"))
        
        if user_filter:
            query = query.where(AuditLog.user_id == uuid.UUID(user_filter))
        
        if date_from:
            query = query.where(AuditLog.created_at >= date_from)
        
        if date_to:
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
    
    async def get_compliance_audit_report(
        self,
        workspace_id: str,
        report_type: str = "monthly"
    ) -> Dict[str, Any]:
        """Generate compliance audit report for German authorities"""
        
        # Calculate date range based on report type
        end_date = datetime.now(timezone.utc)
        if report_type == "weekly":
            start_date = end_date - timedelta(days=7)
        elif report_type == "monthly":
            start_date = end_date - timedelta(days=30)
    async def get_compliance_audit_report(
        self,
        workspace_id: str,
        report_type: str = "monthly"
    ) -> Dict[str, Any]:
        """Generate compliance audit report for German authorities"""
        
        # Calculate date range based on report type
        end_date = datetime.now(timezone.utc)
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
    
    def _determine_gdpr_basis(self, action: str, resource_type: Optional[str]) -> str:
        """Determine GDPR legal basis for processing"""
        
        # Map actions to GDPR Article 6 legal bases
        legal_basis_mapping = {
            # Legitimate interest (Art. 6(1)(f))
            "document_analysis": "Art. 6(1)(f) - Legitimate interest for compliance analysis",
            "compliance_check": "Art. 6(1)(f) - Legitimate interest for regulatory compliance",
            "audit_log_creation": "Art. 6(1)(f) - Legitimate interest for security and audit",
            
            # Contractual necessity (Art. 6(1)(b))
            "user_login": "Art. 6(1)(b) - Contract performance for service provision",
            "workspace_access": "Art. 6(1)(b) - Contract performance for service access",
            
            # Legal obligation (Art. 6(1)(c))
            "audit_trail_maintenance": "Art. 6(1)(c) - Legal obligation for audit requirements",
            "data_retention_enforcement": "Art. 6(1)(c) - Legal obligation for data protection",
            
            # Consent (Art. 6(1)(a)) - for optional features
            "analytics_tracking": "Art. 6(1)(a) - Consent for service improvement",
            "marketing_communication": "Art. 6(1)(a) - Consent for marketing purposes"
        }
        
        return legal_basis_mapping.get(action, "Art. 6(1)(f) - Legitimate interest for service operation")
    
    def _categorize_data_processing(self, action: str, resource_type: Optional[str]) -> str:
        """Categorize type of data being processed"""
        
        # Categorize based on action and resource type
        if action in ["user_login", "user_registration", "profile_update"]:
            return "user_authentication_data"
        elif action in ["document_upload", "document_analysis", "compliance_check"]:
            return "compliance_document_data"
        elif "audit" in action:
            return "audit_log_data"
        elif resource_type == "workspace":
            return "workspace_configuration_data"
        elif resource_type == "analysis":
            return "compliance_analysis_data"
        else:
            return "operational_data"
    
    def _analyze_activity_patterns(self, audit_entries: List[AuditLog]) -> Dict[str, Any]:
        """Analyze activity patterns for security insights"""
        
        activity_summary = {
            "total_actions": len(audit_entries),
            "unique_users": len(set(str(entry.user_id) for entry in audit_entries if entry.user_id)),
            "action_breakdown": {},
            "hourly_distribution": {},
            "success_rate": 0.0,
            "most_active_users": [],
            "peak_activity_periods": []
        }
        
        # Action breakdown
        for entry in audit_entries:
            action = entry.action
            activity_summary["action_breakdown"][action] = activity_summary["action_breakdown"].get(action, 0) + 1
        
        # Success rate
        successful_actions = sum(1 for entry in audit_entries if entry.status == "success")
        activity_summary["success_rate"] = successful_actions / len(audit_entries) if audit_entries else 0.0
        
        # Hourly distribution
        for entry in audit_entries:
            hour = entry.created_at.hour
            activity_summary["hourly_distribution"][hour] = activity_summary["hourly_distribution"].get(hour, 0) + 1
        
        # Most active users (anonymized for privacy)
        user_activity = {}
        for entry in audit_entries:
            if entry.user_id:
                user_key = str(entry.user_id)[:8] + "..."  # Anonymize
                user_activity[user_key] = user_activity.get(user_key, 0) + 1
        
        activity_summary["most_active_users"] = sorted(
            user_activity.items(), key=lambda x: x[1], reverse=True
        )[:5]
        
        return activity_summary
    
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
        
        current_time = datetime.now(timezone.utc)
        
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
                    "timestamp": entry.created_at.isoformat(),
                    "action": entry.action,
                    "error": entry.error_message
                })
        
        return gdpr_analysis
    
    def _identify_security_events(self, audit_entries: List[AuditLog]) -> List[Dict[str, Any]]:
        """Identify security-relevant events for monitoring"""
        
        security_events = []
        
        # Define security-relevant patterns
        security_patterns = {
            "failed_login": ["login_failed", "authentication_failed"],
            "unauthorized_access": ["access_denied", "permission_denied"],
            "suspicious_activity": ["multiple_failures", "rate_limit_exceeded"],
            "data_access": ["document_download", "bulk_export"],
            "admin_actions": ["user_created", "permissions_modified", "workspace_settings_changed"]
        }
        
        for entry in audit_entries:
            for event_type, patterns in security_patterns.items():
                if any(pattern in entry.action for pattern in patterns):
                    security_events.append({
                        "event_type": event_type,
                        "timestamp": entry.created_at.isoformat(),
                        "action": entry.action,
                        "user_id": str(entry.user_id)[:8] + "..." if entry.user_id else "System",
                        "ip_address": entry.ip_address,
                        "status": entry.status,
                        "details": entry.details
                    })
        
        return security_events
    
    def _analyze_user_activity(self, audit_entries: List[AuditLog]) -> Dict[str, Any]:
        """Analyze user activity patterns"""
        
        user_analysis = {
            "active_users_count": 0,
            "user_activity_patterns": {},
            "access_patterns": {},
            "collaboration_metrics": {}
        }
        
        user_activities = {}
        
        for entry in audit_entries:
            if not entry.user_id:
                continue
                
            user_key = str(entry.user_id)
            if user_key not in user_activities:
                user_activities[user_key] = {
                    "total_actions": 0,
                    "action_types": set(),
                    "first_activity": entry.created_at,
                    "last_activity": entry.created_at,
                    "success_rate": 0,
                    "failures": 0
                }
            
            user_data = user_activities[user_key]
            user_data["total_actions"] += 1
            user_data["action_types"].add(entry.action)
            
            if entry.created_at < user_data["first_activity"]:
                user_data["first_activity"] = entry.created_at
            if entry.created_at > user_data["last_activity"]:
                user_data["last_activity"] = entry.created_at
                
            if entry.status == "failure":
                user_data["failures"] += 1
        
        # Calculate metrics
        user_analysis["active_users_count"] = len(user_activities)
        
        for user_key, data in user_activities.items():
            success_rate = (data["total_actions"] - data["failures"]) / data["total_actions"]
            user_analysis["user_activity_patterns"][user_key[:8] + "..."] = {
                "total_actions": data["total_actions"],
                "unique_action_types": len(data["action_types"]),
                "success_rate": success_rate,
                "activity_span_hours": (data["last_activity"] - data["first_activity"]).total_seconds() / 3600
            }
        
        return user_analysis
    
    def _summarize_data_processing(self, audit_entries: List[AuditLog]) -> Dict[str, Any]:
        """Summarize data processing activities for GDPR reporting"""
        
        processing_summary = {
            "total_processing_events": len(audit_entries),
            "data_categories_processed": {},
            "processing_purposes": {},
            "legal_bases_used": {},
            "retention_summary": {
                "entries_approaching_deletion": 0,
                "entries_overdue_deletion": 0
            }
        }
        
        current_time = datetime.now(timezone.utc)
        warning_threshold = current_time + timedelta(days=7)  # 7 days warning
        
        for entry in audit_entries:
            # Data categories
            category = entry.data_category or "unclassified"
            processing_summary["data_categories_processed"][category] = processing_summary["data_categories_processed"].get(category, 0) + 1
            
            # Legal bases
            basis = entry.gdpr_basis or "unspecified"
            processing_summary["legal_bases_used"][basis] = processing_summary["legal_bases_used"].get(basis, 0) + 1
            
            # Processing purposes (inferred from actions)
            purpose = self._infer_processing_purpose(entry.action)
            processing_summary["processing_purposes"][purpose] = processing_summary["processing_purposes"].get(purpose, 0) + 1
            
            # Retention analysis
            if entry.retain_until:
                if entry.retain_until < current_time:
                    processing_summary["retention_summary"]["entries_overdue_deletion"] += 1
                elif entry.retain_until < warning_threshold:
                    processing_summary["retention_summary"]["entries_approaching_deletion"] += 1
        
        return processing_summary
    
    def _infer_processing_purpose(self, action: str) -> str:
        """Infer processing purpose from action"""
        
        purpose_mapping = {
            "document_analysis": "Compliance Analysis",
            "compliance_check": "Regulatory Compliance",
            "user_login": "Service Provision",
            "audit_log": "Security and Audit",
            "workspace_management": "Workspace Administration",
            "data_export": "Data Portability",
            "user_management": "User Administration"
        }
        
        for key, purpose in purpose_mapping.items():
            if key in action:
                return purpose
        
        return "General Service Operation"
    
    def _generate_audit_recommendations(self, audit_entries: List[AuditLog]) -> List[str]:
        """Generate recommendations based on audit analysis"""
        
        recommendations = []
        
        # Analyze failure rate
        total_entries = len(audit_entries)
        failed_entries = sum(1 for entry in audit_entries if entry.status == "failure")
        failure_rate = failed_entries / total_entries if total_entries > 0 else 0
        
        if failure_rate > 0.05:  # More than 5% failure rate
            recommendations.append(
                f"High failure rate detected ({failure_rate:.1%}). Review system stability and user training."
            )
        
        # Check for missing GDPR legal bases
        entries_without_basis = sum(1 for entry in audit_entries if not entry.gdpr_basis)
        if entries_without_basis > 0:
            recommendations.append(
                f"{entries_without_basis} audit entries lack GDPR legal basis documentation. "
                "Review and update action categorization."
            )
        
        # Check retention compliance
        current_time = datetime.now(timezone.utc)
        overdue_entries = sum(1 for entry in audit_entries 
                            if entry.retain_until and entry.retain_until < current_time)
        if overdue_entries > 0:
            recommendations.append(
                f"{overdue_entries} audit entries exceed retention period. "
                "Implement automated deletion process."
            )
        
        # Security recommendations
        security_actions = [entry for entry in audit_entries if entry.status == "failure"]
        if len(security_actions) > 10:
            recommendations.append(
                "Multiple security events detected. Consider implementing additional monitoring and alerting."
            )
        
        # User activity recommendations
        unique_users = len(set(str(entry.user_id) for entry in audit_entries if entry.user_id))
        if total_entries > 0 and unique_users > 0:
            avg_actions_per_user = total_entries / unique_users
            if avg_actions_per_user > 100:
                recommendations.append(
                    "High user activity detected. Consider implementing user activity dashboards for monitoring."
                )
        
        if not recommendations:
            recommendations.append(
                "Audit trail analysis shows good compliance posture. Continue regular monitoring."
            )
        
        return recommendations
    
    async def cleanup_expired_audit_logs(self) -> int:
        """Clean up audit logs that have exceeded retention period"""
        
        current_time = datetime.now(timezone.utc)
        
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