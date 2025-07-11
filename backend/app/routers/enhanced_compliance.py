# app/routers/enhanced_compliance.py
from fastapi import APIRouter, File, UploadFile, Form, HTTPException, Request, Depends, Query
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
from sqlalchemy.ext.asyncio import AsyncSession
import structlog
import time
import asyncio
from dataclasses import dataclass, field

from app.models.compliance import AnalysisResponse, ComplianceFramework, DocumentAnalysis, ComplianceReport, DocumentLanguage
from app.services.file_processor import FileProcessor
from app.services.enhanced_compliance_analyzer import EnhancedComplianceAnalyzer
from app.database import get_db_session, DEMO_WORKSPACE_ID, DEMO_ADMIN_USER_ID
from app.config import settings

# Day 3 Parallel Processing Imports
from app.services.parallel_processing import (
    JobQueue, BatchProcessor, UIContextLayer, PerformanceMonitor
)
from app.services.websocket.progress_handler import progress_handler

# German Authority Engine Import
from app.services.german_authority_engine import GermanAuthorityEngine, get_all_authorities

# Big 4 Authority Engine Integration
from app.services.german_authority_engine.integration.authority_endpoints import (
    Big4AuthorityEndpoints, create_big4_authority_endpoints
)

# Initialize router
router = APIRouter(prefix="/api/v2/compliance", tags=["Day 2 - Enterprise Features with Docling Intelligence"])

logger = structlog.get_logger()

# Initialize Big 4 Engine
big4_endpoints = create_big4_authority_endpoints()

# =============================================================================
# AUTHORITY CONTEXT CLASS - SCOPE FIX IMPLEMENTATION
# =============================================================================

@dataclass
class AuthorityContext:
    """
    Context object to maintain authority detection results throughout processing.
    This ensures authority data persists across async operations and try blocks.
    """
    detected_authority: str = "unknown"
    detected_industry: str = "unknown" 
    authority_confidence: float = 0.0
    authority_analysis: Optional[Any] = None
    authority_guidance: List[str] = field(default_factory=list)
    german_content_detected: bool = False
    
    def has_authority_data(self) -> bool:
        """Check if valid authority data was detected"""
        return self.detected_authority != "unknown" and self.authority_analysis is not None
    
    def to_metadata_dict(self) -> Dict[str, Any]:
        """Convert to metadata dictionary for API response"""
        return {
            "detected_authority": self.detected_authority,
            "detected_industry": self.detected_industry,
            "authority_confidence": self.authority_confidence,
            "authority_analysis_available": self.authority_analysis is not None,
            "authority_guidance_count": len(self.authority_guidance),
            "enforcement_likelihood": self.authority_analysis.enforcement_likelihood if self.authority_analysis else 0.0,
            "penalty_risk_level": self.authority_analysis.penalty_risk_level if self.authority_analysis else "unknown",
            "audit_readiness_score": self.authority_analysis.audit_readiness_score if self.authority_analysis else 0.0
        }

# =============================================================================
# MAIN ANALYSIS ENDPOINT WITH AUTHORITY INTELLIGENCE
# =============================================================================

@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_compliance_with_enterprise_features(
    request: Request,
    files: List[UploadFile] = File(...),
    framework: str = Form(default="gdpr"),
    workspace_id: str = Form(default=DEMO_WORKSPACE_ID),
    user_id: str = Form(default=DEMO_ADMIN_USER_ID),
    company_location: Optional[str] = Form(None),
    industry_hint: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_db_session)
):
    """
    Day 3 Enhanced: 10x Performance + UI Context Intelligence + Authority Detection
    - Parallel processing with intelligent job prioritization
    - Real-time WebSocket progress updates
    - UI context detection for smart interface automation
    - Performance monitoring with German compliance optimization
    - Integrated German authority detection and analysis
    """
    
    start_time = datetime.now(timezone.utc)
    
    # Initialize authority context object to maintain scope throughout processing
    authority_ctx = AuthorityContext()
    
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
    
    # Enhanced file validation
    if len(files) > settings.max_files_per_batch:
        raise HTTPException(
            status_code=400,
            detail=f"Maximum {settings.max_files_per_batch} files allowed per batch analysis"
        )
    
    try:
        # Process and validate files
        processed_files = []
        total_size = 0
        
        for file in files:
            if not any(file.filename.lower().endswith(ext) for ext in settings.allowed_extensions):
                raise HTTPException(
                    status_code=400,
                    detail=f"File type not supported: {file.filename}. "
                           f"Allowed: {', '.join(settings.allowed_extensions)}"
                )
            
            if file.size and file.size > settings.max_file_size_bytes:
                raise HTTPException(
                    status_code=413,
                    detail=f"File {file.filename} exceeds maximum size of {settings.max_file_size_mb}MB"
                )
            
            content = await file.read()
            file_size = len(content)
            total_size += file_size
            processed_files.append((file.filename, content, file_size))
        
        if total_size > settings.max_workspace_size_mb * 1024 * 1024:
            raise HTTPException(
                status_code=400,
                detail=f"Total batch size exceeds workspace limit of {settings.max_workspace_size_mb}MB"
            )
        
        logger.info(
            "Day 3 enhanced processing started",
            workspace_id=workspace_id,
            file_count=len(processed_files),
            total_size_mb=total_size / (1024 * 1024),
            framework=framework
        )
        
        # =================================================================
        # BIG 4 GERMAN AUTHORITY DETECTION WITH SCOPE FIX
        # =================================================================
        
        # Initialize UI Context Layer for content detection
        ui_context_layer = UIContextLayer()
        
        # German content detection
        authority_ctx.german_content_detected = any(
            ui_context_layer._detect_german_content(content) 
            for _, content, _ in processed_files
        )
        
        # Only activate Big 4 Authority Engine for German GDPR content
        if authority_ctx.german_content_detected and str(compliance_framework).lower() == 'gdpr':
            try:
                logger.info("🔍 Big 4 German Authority Engine: Activating for German GDPR content")
                
                # Create document objects for Big 4 engine
                documents = []
                for filename, content, size in processed_files:
                    content_str = content if isinstance(content, str) else content.decode('utf-8', errors='ignore')
                    
                    doc = type('Document', (), {
                        'filename': filename,
                        'content': content_str,
                        'file_size': size,
                        'upload_timestamp': datetime.now(timezone.utc)
                    })()
                    documents.append(doc)
                
                # Use optimized authority detection to maintain scope
                authority_ctx = await _optimized_authority_detection(
                    documents, company_location, industry_hint
                )
                
                logger.info(
                    f"Authority detection completed: {authority_ctx.detected_authority} "
                    f"(industry: {authority_ctx.detected_industry}, "
                    f"confidence: {authority_ctx.authority_confidence:.2f})"
                )
                
            except Exception as e:
                logger.warning(f"Big 4 Authority Engine error (non-critical): {str(e)}")
                # Authority context remains with default values, processing continues
        
        # =================================================================
        # EXISTING ANALYSIS PIPELINE WITH AUTHORITY CONTEXT
        # =================================================================
        
        # Initialize Day 3 processing components
        job_queue = JobQueue()
        batch_processor = BatchProcessor()
        performance_monitor = PerformanceMonitor()
        
        # Generate UI context intelligence
        ui_context = ui_context_layer.analyze_ui_context([])  # Will be populated with jobs
        
        # Send UI context to frontend
        await progress_handler.handle_ui_context_update(workspace_id, ui_context.to_dict())
        
        # Create intelligent job queue with authority context
        jobs = []
        for i, (filename, content, size) in enumerate(processed_files):
            job_data = {
                "job_id": f"{workspace_id}_{i}_{int(time.time())}",
                "workspace_id": workspace_id,
                "user_id": user_id,
                "filename": filename,
                "content": content,
                "file_size": size,
                "framework": compliance_framework,
                "priority": 1000 if authority_ctx.german_content_detected else 500,
                
                # Pass the entire authority context to maintain scope
                "authority_context": authority_ctx,
                
                "optimization_applied": True,
                "is_german_compliance": authority_ctx.german_content_detected
            }
            
            # Create job object
            job = DocumentJob(
                job_id=job_data["job_id"],
                workspace_id=workspace_id,
                user_id=user_id,
                filename=filename,
                content=content,
                size=size,
                framework=compliance_framework,
                priority=job_data["priority"],
                is_german_compliance=authority_ctx.german_content_detected,
                authority_context=authority_ctx
            )
            jobs.append(job)
        
        # Create intelligent batches for parallel processing
        batches = job_queue.create_intelligent_batches(jobs)
        
        # Start performance monitoring
        performance_monitor.start_batch_monitoring(len(jobs))
        
        # Notify frontend of batch start
        await progress_handler.handle_batch_started(workspace_id, {
            "total_jobs": len(jobs),
            "total_batches": len(batches),
            "german_docs": sum(1 for job in jobs if job.is_german_compliance),
            "framework": framework,
            "ui_context": ui_context.to_dict(),
            "authority_intelligence": authority_ctx.to_metadata_dict()
        })
        
        # Create progress callback for real-time updates
        progress_callback = progress_handler.create_progress_callback(workspace_id)
        
        # Process batches with parallel intelligence
        processing_results = await batch_processor.process_batches(
            batches, workspace_id, user_id, compliance_framework, progress_callback
        )
        
        # Record performance metrics
        performance_monitor.record_processing_results(processing_results)
        
        # Convert processing results to document analyses
        individual_analyses = []
        for result in processing_results:
            if result.success and result.analysis:
                # Enhance analysis with authority data if available
                if authority_ctx.has_authority_data():
                    if hasattr(result.analysis, 'german_insights') and result.analysis.german_insights:
                        result.analysis.german_insights["detected_authority"] = authority_ctx.detected_authority
                        result.analysis.german_insights["detected_industry"] = authority_ctx.detected_industry
                        result.analysis.german_insights["authority_guidance"] = authority_ctx.authority_guidance[:2]
                
                individual_analyses.append(result.analysis)
            else:
                # Create error analysis for failed processing
                individual_analyses.append(_create_error_analysis(result))
        
        # Create enhanced compliance report with authority context
        compliance_report = await _create_enhanced_compliance_report(
            individual_analyses, ui_context, performance_monitor, compliance_framework, workspace_id,
            authority_context=authority_ctx
        )
        
        processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()
        
        # =================================================================
        # ENHANCED RESPONSE WITH AUTHORITY INTELLIGENCE
        # =================================================================
        
        # Enhanced executive summary with authority information
        enhanced_summary = compliance_report.executive_summary
        
        if authority_ctx.has_authority_data():
            authority_name = authority_ctx.authority_analysis.authority_name
            enhanced_summary += f"\n\nGerman Authority: {authority_name}"
            
            if authority_ctx.detected_industry != "unknown":
                enhanced_summary += f"\nIndustry: {authority_ctx.detected_industry.title()}"
            
            if authority_ctx.authority_guidance:
                enhanced_summary += f"\nAuthority-Specific Requirements: {len(authority_ctx.authority_guidance)} identified"
        
        # Enhanced next steps with authority guidance
        enhanced_next_steps = compliance_report.next_steps.copy()
        if authority_ctx.authority_guidance:
            enhanced_next_steps = authority_ctx.authority_guidance + enhanced_next_steps
        
        # Create comprehensive response with Day 3 enhancements
        analysis_response = AnalysisResponse(
            individual_analyses=individual_analyses,
            compliance_report=ComplianceReport(
                framework=compliance_report.framework,
                executive_summary=enhanced_summary,
                executive_summary_de=getattr(compliance_report, 'executive_summary_de', None),
                compliance_score=compliance_report.compliance_score,
                documents_analyzed=compliance_report.documents_analyzed,
                german_documents_detected=compliance_report.german_documents_detected,
                priority_gaps=compliance_report.priority_gaps,
                compliance_strengths=compliance_report.compliance_strengths,
                next_steps=enhanced_next_steps,
                german_specific_recommendations=compliance_report.german_specific_recommendations
            ),
            processing_metadata={
                "analysis_id": f"day3_{int(start_time.timestamp())}",
                "processing_time": processing_time,
                "total_documents": len(processed_files),
                "total_batches": len(batches),
                "framework": framework,
                "workspace_id": workspace_id,
                "performance_grade": _calculate_performance_grade(processing_results),
                "day3_features": {
                    "parallel_processing": True,
                    "ui_context_intelligence": True,
                    "german_priority_processing": True,
                    "real_time_progress": True,
                    "performance_monitoring": True,
                    "authority_intelligence": True,
                    "performance_optimization": True
                },
                "ui_context": ui_context.to_dict(),
                "performance_metrics": performance_monitor.get_processing_statistics(),
                
                # Use the context object's metadata method for authority data
                "authority_intelligence": authority_ctx.to_metadata_dict()
            }
        )
        
        # Notify frontend of completion with authority data
        await progress_handler.handle_batch_completed(workspace_id, {
            "documents_analyzed": len(individual_analyses),
            "processing_time": processing_time,
            "compliance_score": compliance_report.compliance_score,
            "german_documents_detected": compliance_report.german_documents_detected,
            "success_rate": len([r for r in processing_results if r.success]) / len(processing_results),
            "performance_grade": _calculate_performance_grade(processing_results),
            "ui_context": ui_context.to_dict(),
            "authority_intelligence": authority_ctx.to_metadata_dict()
        })
        
        logger.info(
            "Day 3 enhanced analysis completed successfully",
            workspace_id=workspace_id,
            processing_time=processing_time,
            documents_processed=len(individual_analyses),
            compliance_score=compliance_report.compliance_score,
            performance_grade=_calculate_performance_grade(processing_results),
            detected_authority=authority_ctx.detected_authority,
            detected_industry=authority_ctx.detected_industry
        )
        
        return analysis_response
        
    except HTTPException:
        raise
    except Exception as e:
        # Enhanced error handling with progress notification
        await progress_handler.handle_error_notification(workspace_id, {
            "error_type": "processing_error",
            "message": str(e),
            "workspace_id": workspace_id,
            "framework": framework
        })
        
        logger.error(
            "Day 3 enhanced analysis failed",
            workspace_id=workspace_id,
            error=str(e)
        )
        
        raise HTTPException(
            status_code=500,
            detail=f"Enhanced compliance analysis failed: {str(e)}"
        )

# =============================================================================
# OPTIMIZED AUTHORITY DETECTION - SCOPE FIX HELPER
# =============================================================================

async def _optimized_authority_detection(
    documents: List[Any],
    company_location: Optional[str],
    industry_hint: Optional[str]
) -> AuthorityContext:
    """
    Optimized authority detection with performance considerations.
    Uses concurrent execution where possible and maintains proper scope.
    """
    context = AuthorityContext()
    
    try:
        from app.services.german_authority_engine.big4.big4_detector import Big4AuthorityDetector
        from app.services.german_authority_engine.big4.big4_analyzer import Big4ComplianceAnalyzer
        
        detector = Big4AuthorityDetector()
        analyzer = Big4ComplianceAnalyzer()
        
        # Run industry detection and authority detection concurrently
        industry_task = detector.detect_industry_from_content(documents)
        
        # Start with industry hint if available to speed up detection
        initial_industry = industry_hint if industry_hint else None
        
        # Detect authorities with initial hint
        detection_task = detector.detect_relevant_authorities(
            documents=documents,
            suggested_industry=initial_industry,
            suggested_state=company_location
        )
        
        # Await both tasks
        detected_industry, detection_result = await asyncio.gather(
            industry_task, detection_task
        )
        
        # Update context with detection results
        context.detected_industry = detected_industry if detected_industry != "unknown" else (industry_hint or "unknown")
        context.german_content_detected = True  # Already confirmed by caller
        
        if detection_result.primary_authority:
            context.detected_authority = detection_result.primary_authority.value
            context.authority_confidence = detection_result.detection_confidence
            
            logger.info(
                f"Authority detected: {context.detected_authority} "
                f"(confidence: {context.authority_confidence:.2f})"
            )
            
            # Perform analysis only if detection confidence is high enough
            if context.authority_confidence >= 0.5:  # 50% threshold
                context.authority_analysis = await analyzer.analyze_for_authority(
                    documents=documents,
                    authority=detection_result.primary_authority,
                    industry=context.detected_industry
                )
                
                # Extract authority-specific guidance
                if context.authority_analysis:
                    context.authority_guidance = [
                        f"{context.authority_analysis.authority_name} requires: {req}" 
                        for req in context.authority_analysis.requirements_missing[:3]
                    ]
                    
                    # Add industry-specific guidance
                    if context.authority_analysis.industry_specific_guidance:
                        context.authority_guidance.extend(
                            context.authority_analysis.industry_specific_guidance[:2]
                        )
                    
                    logger.info(
                        f"Authority analysis completed - "
                        f"{len(context.authority_guidance)} guidance items generated"
                    )
    
    except Exception as e:
        logger.warning(f"Optimized authority detection failed: {str(e)}")
        # Return context with default values
    
    return context

# =============================================================================
# BIG 4 GERMAN AUTHORITY ENDPOINTS - Added for Enhanced Authority Analysis
# =============================================================================

@router.post("/analyze-with-authority-detection")
async def analyze_with_authority_detection(
    files: List[UploadFile] = File(...),
    industry: Optional[str] = Form(None),
    company_location: Optional[str] = Form(None), 
    company_size: Optional[str] = Form(None),
    workspace_id: str = Form(default=DEMO_WORKSPACE_ID),
    db: AsyncSession = Depends(get_db_session)
):
    """
    🧠 Smart Authority Detection + Analysis
    
    Automatically detects relevant Big 4 German authorities and provides
    comprehensive analysis with multi-authority comparison.
    
    Enhanced Features:
    - Intelligent authority detection based on content and business context
    - Multi-authority comparison and optimization recommendations
    - Industry-specific compliance templates
    - Authority-specific penalty risk assessment
    """
    return await big4_endpoints.analyze_with_smart_detection(
        files=files,
        industry=industry,
        company_location=company_location,
        company_size=company_size,
        workspace_id=workspace_id,
        db=db
    )

@router.post("/analyze-authority/{authority_id}")
async def analyze_authority_specific(
    authority_id: str,
    files: List[UploadFile] = File(...),
    industry: Optional[str] = Form(None),
    company_size: Optional[str] = Form(None),
    workspace_id: str = Form(default=DEMO_WORKSPACE_ID),
    db: AsyncSession = Depends(get_db_session)
):
    """
    🎯 Authority-Specific Compliance Analysis
    
    Detailed analysis for specific Big 4 German authority with
    enforcement patterns, penalty estimates, and audit preparation.
    
    Supported Authorities:
    - bfdi: Federal Commissioner (BfDI)
    - baylda: Bavaria (BayLDA) - Automotive focus
    - lfd_bw: Baden-Württemberg (LfD BW) - Software focus  
    - ldi_nrw: North Rhine-Westphalia (LDI NRW) - Manufacturing focus
    """
    return await big4_endpoints.analyze_for_specific_authority(
        authority_id=authority_id,
        files=files,
        industry=industry,
        company_size=company_size,
        workspace_id=workspace_id,
        db=db
    )

@router.post("/compare-authorities")
async def compare_authority_compliance(
    files: List[UploadFile] = File(...),
    authorities: List[str] = Form(...),
    industry: Optional[str] = Form(None),
    company_size: Optional[str] = Form(None),
    workspace_id: str = Form(default=DEMO_WORKSPACE_ID),
    db: AsyncSession = Depends(get_db_session)
):
    """
    ⚖️ Multi-Authority Compliance Comparison
    
    Compare compliance analysis across multiple Big 4 authorities
    to optimize jurisdiction strategy and identify best practices.
    
    Features:
    - Side-by-side authority comparison
    - Jurisdiction optimization recommendations
    - Cost-benefit analysis for compliance strategies
    - Implementation roadmap for optimal compliance
    """
    return await big4_endpoints.compare_authorities(
        files=files,
        authorities=authorities,
        industry=industry,
        company_size=company_size,
        workspace_id=workspace_id,
        db=db
    )

@router.get("/authorities/detect-from-business")
async def detect_authorities_from_business(
    company_location: str = Query(..., description="Company location (e.g., 'bayern', 'baden_wurttemberg')"),
    industry: str = Query(..., description="Industry (e.g., 'automotive', 'software', 'manufacturing')"),
    company_size: str = Query(..., description="Company size ('small', 'medium', 'large')"),
    business_activities: Optional[List[str]] = Query(None, description="Specific business activities")
):
    """
    🔍 Business Profile Authority Detection
    
    Detect relevant German authorities based on business profile
    without requiring document upload. Perfect for onboarding.
    
    Use Cases:
    - Initial authority identification during onboarding
    - Strategic jurisdiction planning
    - Compliance planning before document creation
    """
    return await big4_endpoints.detect_relevant_authorities(
        company_location=company_location,
        industry=industry,
        company_size=company_size,
        business_activities=business_activities
    )

@router.get("/templates/industry/{industry}")
async def get_industry_compliance_template(
    industry: str,
    authority: Optional[str] = Query(None, description="Specific authority for customized template")
):
    """
    📋 Industry-Specific Compliance Templates
    
    Get pre-configured compliance templates for German industries
    with authority-specific requirements and best practices.
    
    Supported Industries:
    - automotive: Connected vehicles, supplier agreements
    - software: Privacy by design, API compliance
    - manufacturing: IoT compliance, employee monitoring
    - healthcare: Patient data, medical research
    """
    return await big4_endpoints.get_industry_template(
        industry=industry,
        authority=authority
    )

@router.get("/authorities/big4")
async def get_big4_authorities():
    """
    🏛️ Big 4 German Authorities Information
    
    Complete information about the Big 4 German data protection authorities
    including enforcement patterns, contact information, and specializations.
    
    Coverage:
    - 70% of German SME market
    - All major German business centers
    - Industry-specific enforcement expertise
    """
    return await big4_endpoints.get_all_big4_authorities_info()

@router.get("/german-authorities")
async def get_german_authorities():
    """Get all German data protection authorities"""
    return {
        "authorities": [
            {"id": k.value, "name": v.name} 
            for k, v in get_all_authorities().items()
        ]
    }

# =============================================================================
# HELPER FUNCTIONS WITH AUTHORITY CONTEXT SUPPORT
# =============================================================================

def _create_error_analysis(result) -> DocumentAnalysis:
    """Create error analysis for failed processing result"""
    
    return DocumentAnalysis(
        filename=result.job.filename,
        document_language=DocumentLanguage.UNKNOWN,
        compliance_summary=f"Processing failed: {result.error_message}",
        control_mappings=[],
        compliance_gaps=["Processing error prevented analysis"],
        risk_indicators=["Document processing failed"],
        german_insights=None,
        original_size=result.job.size,
        processing_time=result.processing_time
    )

async def _create_enhanced_compliance_report(
    analyses: List[DocumentAnalysis],
    ui_context,  # UIContext type
    performance_monitor,  # PerformanceMonitor type
    framework: ComplianceFramework,
    workspace_id: str,
    authority_context: Optional[AuthorityContext] = None
) -> ComplianceReport:
    """Create enhanced compliance report with Day 3 intelligence and authority data"""
    
    # Calculate enhanced metrics
    german_documents = sum(
        1 for analysis in analyses 
        if analysis.document_language == DocumentLanguage.GERMAN
    )
    
    # Use UI context for smarter scoring
    base_compliance_score = ui_context.portfolio_score
    
    # Adjust score based on detected scenario and completeness
    scenario_bonus = 0.1 if ui_context.detected_scenario.value != "unknown" else 0.0
    completeness_bonus = ui_context.compliance_completeness * 0.2
    
    # Authority-specific scoring adjustments
    authority_bonus = 0.0
    if authority_context and authority_context.has_authority_data():
        # Higher bonus for higher authority compliance scores
        authority_bonus = authority_context.authority_analysis.compliance_score * 0.1
    
    final_compliance_score = min(1.0, base_compliance_score + scenario_bonus + completeness_bonus + authority_bonus)
    
    # Enhanced executive summary with authority information
    executive_summary = f"""
Day 3 Enhanced Analysis: {framework.value.upper()} compliance assessment completed.

Portfolio Analysis:
- Documents Analyzed: {len(analyses)}
- German Content: {german_documents} documents ({(german_documents/len(analyses)*100):.0f}%)
- Compliance Score: {final_compliance_score:.2%}

Scenario Detected: {ui_context.scenario_description}
Industry: {ui_context.industry_detected.value.title()}
    """.strip()
    
    # Add authority information if available
    if authority_context and authority_context.has_authority_data():
        executive_summary += f"""

Authority Intelligence:
- Detected Authority: {authority_context.authority_analysis.authority_name} ({authority_context.detected_authority})
- Industry: {authority_context.detected_industry.title()}
- Enforcement Likelihood: {authority_context.authority_analysis.enforcement_likelihood:.2%}
- Audit Readiness: {authority_context.authority_analysis.audit_readiness_score:.2%}
        """.strip()
    
    # Enhanced recommendations based on UI context and authority data
    next_steps = [
        action.description for action in ui_context.suggested_actions[:3]
    ]
    
    # Add authority-specific guidance
    if authority_context and authority_context.authority_guidance:
        next_steps = authority_context.authority_guidance + next_steps
    
    next_steps.extend([
        "Review detailed document-level analysis results",
        "Export compliance report for stakeholder review"
    ])
    
    return ComplianceReport(
        framework=framework,
        executive_summary=executive_summary,
        compliance_score=final_compliance_score,
        documents_analyzed=len(analyses),
        german_documents_detected=german_documents > 0,
        priority_gaps=ui_context.priority_risks,
        compliance_strengths=[
            f"Intelligent processing completed in high-performance mode",
            f"UI context detection: {ui_context.detected_scenario.value.replace('_', ' ').title()}",
            f"German compliance optimization active",
            f"Real-time progress tracking enabled"
        ],
        next_steps=next_steps,
        german_specific_recommendations=ui_context.quick_wins if german_documents > 0 else []
    )

def _calculate_performance_grade(results) -> str:
    """Calculate overall performance grade"""
    
    if not results:
        return "F"
    
    successful_results = [r for r in results if r.success]
    success_rate = len(successful_results) / len(results)
    
    if successful_results:
        avg_time = sum(r.processing_time for r in successful_results) / len(successful_results)
    else:
        avg_time = float('inf')
    
    # Grade based on Day 3 targets
    if avg_time <= 3.0 and success_rate >= 0.95:
        return "A"
    elif avg_time <= 5.0 and success_rate >= 0.90:
        return "B"
    elif avg_time <= 8.0 and success_rate >= 0.80:
        return "C"
    elif avg_time <= 12.0 and success_rate >= 0.70:
        return "D"
    else:
        return "F"

# Keep existing endpoints for backward compatibility
@router.get("/health")
async def enhanced_health_check():
    """Enhanced health check for Day 2 enterprise features"""
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "day3_features": {
            "parallel_processing": True,
            "ui_context_intelligence": True,
            "performance_monitoring": True,
            "websocket_progress": True,
            "german_authority_analysis": True,
            "big4_authority_engine": True,
            "authority_scope_fix": True
        }
    }

@router.get("/frameworks")
async def get_supported_frameworks():
    """Get list of supported compliance frameworks with Day 3 enhancements"""
    return {
        "supported_frameworks": [
            {
                "id": "gdpr",
                "name": "DSGVO / GDPR",
                "description": "EU-Datenschutzgrundverordnung / EU Data Protection Regulation",
                "region": "European Union",
                "german_support": True,
                "day3_features": {
                    "parallel_processing": True,
                    "ui_context_detection": True,
                    "german_priority": True,
                    "authority_specific_analysis": True,
                    "big4_authority_engine": True,
                    "authority_scope_fix": True
                }
            },
            {
                "id": "soc2",
                "name": "SOC 2",
                "description": "Security, availability, and confidentiality controls",
                "region": "Global (US-originated)",
                "german_support": False,
                "day3_features": {
                    "parallel_processing": True,
                    "ui_context_detection": True,
                    "german_priority": False,
                    "authority_specific_analysis": False,
                    "big4_authority_engine": False,
                    "authority_scope_fix": False
                }
            },
            {
                "id": "hipaa",
                "name": "HIPAA",
                "description": "Healthcare information privacy and security",
                "region": "United States",
                "german_support": False,
                "day3_features": {
                    "parallel_processing": True,
                    "ui_context_detection": True,
                    "german_priority": False,
                    "authority_specific_analysis": False,
                    "big4_authority_engine": False,
                    "authority_scope_fix": False
                }
            },
            {
                "id": "iso27001",
                "name": "ISO 27001",
                "description": "Information security management systems",
                "region": "Global",
                "german_support": False,
                "day3_features": {
                    "parallel_processing": True,
                    "ui_context_detection": True,
                    "german_priority": False,
                    "authority_specific_analysis": False,
                    "big4_authority_engine": False,
                    "authority_scope_fix": False
                }
            }
        ]
    }