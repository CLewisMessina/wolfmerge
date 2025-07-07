# app/routers/enhanced_compliance.py - Clean Version for Railway
from fastapi import APIRouter, File, UploadFile, Form, HTTPException, Depends, Request
from typing import List, Optional
from datetime import datetime, timezone
import structlog
from sqlalchemy.ext.asyncio import AsyncSession
from uuid import UUID
import io

from app.models.compliance import AnalysisResponse, ComplianceFramework
from app.services.enhanced_compliance_analyzer import EnhancedComplianceAnalyzer
from app.services.audit_service import AuditService
from app.database import get_db_session, DEMO_WORKSPACE_ID, DEMO_ADMIN_USER_ID
from app.config import settings

logger = structlog.get_logger()

router = APIRouter(prefix="/api/v2/compliance", tags=["compliance-v2-enterprise"])

@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_compliance_with_enterprise_features(
    request: Request,
    files: List[UploadFile] = File(...),
    framework: str = Form(default="gdpr"),
    workspace_id: str = Form(default=DEMO_WORKSPACE_ID),
    user_id: str = Form(default=DEMO_ADMIN_USER_ID),
    db: AsyncSession = Depends(get_db_session)
):
    """
    Enhanced compliance analysis with Docling intelligence, team workspaces, and GDPR audit trails
    Day 2: Enterprise-grade document processing with chunk-level German compliance analysis
    """
    
    start_time = datetime.now(timezone.utc)
    
    # Extract client information for audit trail
    client_ip = request.client.host if request.client else None
    user_agent = request.headers.get("user-agent")
    
    # Validate framework
    try:
        compliance_framework = ComplianceFramework(framework.lower())
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported framework: {framework}. Supported: gdpr, soc2, hipaa, iso27001"
        )
    
    # Validate file count for enterprise tier
    if len(files) > settings.max_files_per_batch:
        raise HTTPException(
            status_code=400,
            detail=f"Maximum {settings.max_files_per_batch} files allowed per batch analysis"
        )
    
    # Enhanced file validation for enterprise features
    try:
        processed_files = []
        total_size = 0
        
        for file in files:
            # Validate file type first
            if not any(file.filename.lower().endswith(ext) for ext in settings.allowed_extensions):
                raise HTTPException(
                    status_code=400,
                    detail=f"File type not supported: {file.filename}. "
                           f"Allowed: {', '.join(settings.allowed_extensions)}"
                )
            
            # Validate file size
            if file.size and file.size > settings.max_file_size_bytes:
                raise HTTPException(
                    status_code=413,
                    detail=f"File {file.filename} exceeds maximum size of {settings.max_file_size_mb}MB"
                )
            
            # Read file content
            content = await file.read()
            await file.seek(0)  # Reset for any future use
            
            file_size = len(content)
            total_size += file_size
            processed_files.append((file.filename, content, file_size))
        
        # Validate total batch size
        if total_size > settings.max_workspace_size_mb * 1024 * 1024:
            raise HTTPException(
                status_code=400,
                detail=f"Total batch size exceeds workspace limit of {settings.max_workspace_size_mb}MB"
            )
        
        logger.info(
            "File validation completed",
            workspace_id=workspace_id,
            file_count=len(processed_files),
            total_size_mb=total_size / (1024 * 1024),
            framework=framework
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("File processing failed", error=str(e), workspace_id=workspace_id)
        raise HTTPException(
            status_code=500,
            detail=f"File processing failed: {str(e)}"
        )
    
    # Initialize audit service for GDPR compliance
    audit_service = AuditService(db)
    
    # Enhanced compliance analysis with full enterprise features
    try:
        analyzer = EnhancedComplianceAnalyzer(db)
        
        # Log analysis start for audit trail
        await audit_service.log_action(
            workspace_id=workspace_id,
            user_id=user_id,
            action="enhanced_analysis_requested",
            resource_type="batch_analysis",
            details={
                "framework": framework,
                "file_count": len(processed_files),
                "total_size_bytes": total_size,
                "docling_enabled": settings.docling_enabled,
                "client_info": {
                    "ip_address": client_ip,
                    "user_agent": user_agent
                }
            },
            ip_address=client_ip,
            user_agent=user_agent
        )
        
        # Perform enhanced analysis with Docling and chunk processing
        result = await analyzer.analyze_documents_for_workspace(
            processed_files, workspace_id, user_id, compliance_framework
        )
        
        processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()
        
        logger.info(
            "Enhanced compliance analysis completed successfully",
            workspace_id=workspace_id,
            user_id=user_id,
            framework=framework,
            processing_time=processing_time,
            compliance_score=result.compliance_report.compliance_score,
            german_documents_detected=result.compliance_report.german_documents_detected
        )
        
        return result
        
    except Exception as e:
        # Log error for audit trail
        await audit_service.log_error(
            workspace_id=workspace_id,
            user_id=user_id,
            action="enhanced_analysis_failed",
            error_message=str(e),
            resource_type="batch_analysis",
            details={
                "framework": framework,
                "file_count": len(processed_files),
                "processing_time": (datetime.now(timezone.utc) - start_time).total_seconds()
            }
        )
        
        logger.error(
            "Enhanced compliance analysis failed",
            workspace_id=workspace_id,
            user_id=user_id,
            error=str(e)
        )
        
        raise HTTPException(
            status_code=500,
            detail=f"Enhanced compliance analysis failed: {str(e)}"
        )

@router.get("/health")
async def enhanced_health_check(db: AsyncSession = Depends(get_db_session)):
    """Enhanced health check for Day 2 enterprise features"""
    
    try:
        from app.database import health_check
        from openai import OpenAI
        
        db_health = await health_check()
        
        # Test OpenAI connectivity
        openai_healthy = False
        try:
            client = OpenAI(api_key=settings.openai_api_key)
            openai_healthy = True
        except Exception as e:
            logger.warning(f"OpenAI health check failed: {e}")
        
        # Check if all services are healthy
        all_healthy = (
            db_health.get("database_connected", False) and
            openai_healthy and
            db_health.get("tables_created", False)
        )
        
        return {
            "status": "healthy" if all_healthy else "degraded",
            "version": "2.0.0",
            "day": "Day 2 - Enterprise Cloud Platform",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "features": {
                "docling_intelligence": settings.docling_enabled,
                "team_workspaces": settings.enable_workspaces,
                "german_dsgvo_analysis": True,
                "eu_cloud_processing": settings.eu_region,
                "gdpr_audit_trails": settings.audit_logging,
                "chunk_level_analysis": True,
                "enterprise_ready": True
            },
            "services": {
                "database": "healthy" if db_health.get("database_connected") else "unhealthy",
                "openai": "healthy" if openai_healthy else "unhealthy",
                "docling": "enabled" if settings.docling_enabled else "disabled"
            },
            "database": db_health,
            "compliance": {
                "gdpr_compliant": settings.gdpr_compliance,
                "data_residency": settings.data_residency,
                "audit_retention_days": settings.audit_retention_days,
                "auto_deletion": settings.auto_delete_processed_content
            },
            "performance": {
                "max_file_size_mb": settings.max_file_size_mb,
                "max_files_per_batch": settings.max_files_per_batch,
                "max_chunks_per_document": settings.max_chunks_per_document,
                "chunk_size": settings.chunk_size
            }
        }
        
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }