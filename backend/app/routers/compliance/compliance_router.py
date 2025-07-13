# app/routers/compliance/compliance_router.py
"""
Compliance Router

Pure FastAPI routing layer for compliance analysis endpoints.
This router handles HTTP requests, parameter validation, and response formatting
while delegating all business logic to the ComplianceOrchestrator.
"""

from typing import List, Optional
from fastapi import APIRouter, File, UploadFile, Form, HTTPException, Request, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
import structlog

from app.database import get_db_session, DEMO_WORKSPACE_ID, DEMO_ADMIN_USER_ID
from app.models.compliance import AnalysisResponse
from .compliance_orchestrator import ComplianceOrchestrator

logger = structlog.get_logger()

# Initialize router
router = APIRouter(
    prefix="/api/v2/compliance",
    tags=["Compliance Analysis - Refactored"]
)

def get_orchestrator(db_session: AsyncSession = Depends(get_db_session)) -> ComplianceOrchestrator:
    """Dependency injection for ComplianceOrchestrator"""
    return ComplianceOrchestrator(db_session)

@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_documents(
    files: List[UploadFile] = File(..., description="Documents to analyze (PDF, DOCX, TXT, etc.)"),
    framework: str = Form("gdpr", description="Compliance framework (gdpr, soc2, hipaa, iso27001)"),
    workspace_id: str = Form(DEMO_WORKSPACE_ID, description="Workspace ID for team collaboration"),
    user_id: str = Form(DEMO_ADMIN_USER_ID, description="User ID for audit trail"),
    company_location: Optional[str] = Form(None, description="Company location for authority detection (e.g., 'Bayern, Germany')"),
    industry_hint: Optional[str] = Form(None, description="Industry hint for specialized analysis (automotive, healthcare, etc.)"),
    request: Request = None,
    orchestrator: ComplianceOrchestrator = Depends(get_orchestrator)
):
    """
    🚀 **Enhanced Compliance Analysis with German Authority Intelligence**
    
    Perform comprehensive compliance analysis with intelligent German authority detection,
    parallel processing, and real-time progress tracking.
    
    ## Features
    - **Multi-framework Support**: GDPR/DSGVO, SOC 2, HIPAA, ISO 27001
    - **German Authority Detection**: Automatic BfDI, BayLDA, LfD detection
    - **Intelligent Document Processing**: Docling integration with semantic chunking
    - **Real-time Progress**: WebSocket updates during processing
    - **Performance Monitoring**: A-F grading with optimization recommendations
    - **Team Workspaces**: Collaborative compliance management
    
    ## German Authority Intelligence
    - **BfDI**: Federal Commissioner for Data Protection (international transfers)
    - **BayLDA**: Bavarian State Office (automotive industry focus)
    - **LfD BW**: Baden-Württemberg Commissioner (manufacturing focus)
    - **Industry-Specific**: Automotive, healthcare, manufacturing compliance
    
    ## Performance Targets
    - **Processing Speed**: < 3 seconds per document
    - **Batch Processing**: Up to 20 documents simultaneously
    - **Success Rate**: > 95% processing reliability
    - **Real-time Updates**: Progress every 0.5 seconds
    """
    
    # Validate request
    if not files:
        raise HTTPException(
            status_code=400,
            detail="No files provided. Please upload at least one document for analysis."
        )
    
    # Extract client information for audit trail
    client_ip = request.client.host if request and request.client else None
    user_agent = request.headers.get("user-agent") if request else None
    
    logger.info(
        "Compliance analysis request received",
        workspace_id=workspace_id,
        user_id=user_id,
        file_count=len(files),
        framework=framework,
        company_location=company_location,
        industry_hint=industry_hint,
        client_ip=client_ip
    )
    
    try:
        # Delegate to orchestrator for processing
        result = await orchestrator.process_compliance_analysis(
            files=files,
            framework=framework,
            workspace_id=workspace_id,
            user_id=user_id,
            company_location=company_location,
            industry_hint=industry_hint,
            client_ip=client_ip,
            user_agent=user_agent
        )
        
        logger.info(
            "Compliance analysis completed successfully",
            workspace_id=workspace_id,
            documents_analyzed=len(result.individual_analyses),
            compliance_score=result.compliance_report.compliance_score,
            framework=framework
        )
        
        return result
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(
            "Compliance analysis failed",
            workspace_id=workspace_id,
            user_id=user_id,
            error=str(e),
            exc_info=True
        )
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )

@router.get("/session/{session_id}/status")
async def get_session_status(
    session_id: str,
    orchestrator: ComplianceOrchestrator = Depends(get_orchestrator)
):
    """
    📊 **Get Real-time Session Status**
    
    Monitor the progress of an active compliance analysis session
    with real-time performance metrics and phase tracking.
    """
    
    try:
        status = await orchestrator.get_session_status(session_id)
        return status
    except Exception as e:
        logger.error(f"Failed to get session status: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve session status: {str(e)}"
        )

@router.get("/performance/summary")
async def get_performance_summary(
    orchestrator: ComplianceOrchestrator = Depends(get_orchestrator)
):
    """
    📈 **Performance Analytics Dashboard**
    
    Get comprehensive performance analytics including:
    - Average processing times and grades
    - Common bottlenecks and optimization opportunities
    - Success rates and throughput metrics
    - Historical performance trends
    """
    
    try:
        summary = await orchestrator.get_performance_summary()
        return summary
    except Exception as e:
        logger.error(f"Failed to get performance summary: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve performance summary: {str(e)}"
        )

@router.get("/health")
async def health_check(
    orchestrator: ComplianceOrchestrator = Depends(get_orchestrator)
):
    """
    🏥 **System Health Check**
    
    Comprehensive health check of all compliance system components:
    - File processing service
    - Authority detection service  
    - Performance monitoring
    - Response builder
    - Database connectivity
    """
    
    try:
        health = await orchestrator.health_check()
        
        # Return appropriate HTTP status based on health
        if health["orchestrator"] == "unhealthy":
            raise HTTPException(
                status_code=503,
                detail=health
            )
        elif health["orchestrator"] == "degraded":
            # Return 200 but indicate degraded performance
            health["warning"] = "Some components are unhealthy but core functionality is available"
        
        return health
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Health check failed: {str(e)}"
        )

@router.get("/frameworks")
async def get_supported_frameworks():
    """
    📋 **Supported Compliance Frameworks**
    
    Get list of all supported compliance frameworks with descriptions
    and German-specific features.
    """
    
    return {
        "frameworks": [
            {
                "id": "gdpr",
                "name": "GDPR/DSGVO",
                "description": "General Data Protection Regulation with German authority intelligence",
                "features": [
                    "German authority detection (BfDI, BayLDA, LfD)",
                    "DSGVO article mapping",
                    "German legal terminology recognition",
                    "Industry-specific compliance (automotive, healthcare)",
                    "Cross-border transfer analysis"
                ]
            },
            {
                "id": "soc2",
                "name": "SOC 2",
                "description": "Service Organization Control 2 framework",
                "features": [
                    "Trust Services Criteria mapping",
                    "Security control assessment",
                    "Operational effectiveness evaluation"
                ]
            },
            {
                "id": "hipaa", 
                "name": "HIPAA",
                "description": "Health Insurance Portability and Accountability Act",
                "features": [
                    "Healthcare data protection analysis",
                    "PHI identification and controls",
                    "Administrative safeguards assessment"
                ]
            },
            {
                "id": "iso27001",
                "name": "ISO 27001",
                "description": "Information Security Management System standard",
                "features": [
                    "Security control framework mapping",
                    "Risk management assessment",
                    "ISMS documentation analysis"
                ]
            }
        ]
    }

@router.get("/authorities/german")
async def get_german_authorities():
    """
    🏛️ **German Data Protection Authorities**
    
    Get comprehensive information about German data protection authorities
    including specializations, enforcement patterns, and contact information.
    """
    
    return {
        "authorities": [
            {
                "id": "bfdi",
                "name": "BfDI - Bundesbeauftragte für den Datenschutz und die Informationsfreiheit",
                "name_en": "Federal Commissioner for Data Protection and Freedom of Information",
                "specialization": "International data transfers, cross-border processing",
                "jurisdiction": "Federal level, international transfers",
                "enforcement_style": "Technical and procedural focus",
                "website": "https://www.bfdi.bund.de"
            },
            {
                "id": "baylda",
                "name": "BayLDA - Bayerisches Landesamt für Datenschutzaufsicht",
                "name_en": "Bavarian State Office for Data Protection Supervision",
                "specialization": "Automotive industry, manufacturing, technology companies",
                "jurisdiction": "Bavaria (Bayern)",
                "enforcement_style": "Industry-aware, technical audits",
                "website": "https://www.lda.bayern.de"
            },
            {
                "id": "lfd_bw",
                "name": "LfDI BW - Landesbeauftragte für den Datenschutz und die Informationsfreiheit Baden-Württemberg",
                "name_en": "Baden-Württemberg Commissioner for Data Protection and Freedom of Information",
                "specialization": "Manufacturing, automotive, engineering companies",
                "jurisdiction": "Baden-Württemberg",
                "enforcement_style": "Risk-based approach, comprehensive documentation requirements",
                "website": "https://www.baden-wuerttemberg.datenschutz.de"
            }
        ],
        "detection_features": [
            "Automatic authority identification based on company location",
            "Industry-specific authority matching",
            "Content analysis for authority jurisdiction determination",
            "Enforcement pattern analysis and risk assessment"
        ]
    }

# Legacy endpoint compatibility
@router.post("/analyze-legacy")
async def analyze_documents_legacy(
    files: List[UploadFile] = File(...),
    framework: str = Form("gdpr"),
    workspace_id: str = Form(DEMO_WORKSPACE_ID),
    user_id: str = Form(DEMO_ADMIN_USER_ID),
    request: Request = None,
    orchestrator: ComplianceOrchestrator = Depends(get_orchestrator)
):
    """
    🔄 **Legacy Compatibility Endpoint**
    
    Maintains compatibility with existing integrations.
    Redirects to the new analyze endpoint with default parameters.
    """
    
    logger.info("Legacy endpoint used, redirecting to new analyze endpoint")
    
    return await analyze_documents(
        files=files,
        framework=framework,
        workspace_id=workspace_id,
        user_id=user_id,
        request=request,
        orchestrator=orchestrator
    )