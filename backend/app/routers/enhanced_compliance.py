# ADD TO TOP OF app/routers/enhanced_compliance.py

# Add these imports after existing imports:
from app.services.parallel_processing import (
    JobQueue, BatchProcessor, UIContextLayer, PerformanceMonitor
)
from app.services.websocket.progress_handler import progress_handler

# REPLACE the analyze_compliance_with_enterprise_features function with this:

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
    Day 3 Enhanced: 10x Performance + UI Context Intelligence
    - Parallel processing with intelligent job prioritization
    - Real-time WebSocket progress updates
    - UI context detection for smart interface automation
    - Performance monitoring with German compliance optimization
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
        
        # Initialize Day 3 processing components
        job_queue = JobQueue()
        batch_processor = BatchProcessor()
        ui_context_layer = UIContextLayer()
        performance_monitor = PerformanceMonitor()
        
        # Create intelligent job queue with German priority
        jobs = job_queue.create_job_queue(processed_files, workspace_id)
        
        # Generate UI context intelligence
        ui_context = ui_context_layer.analyze_ui_context(jobs)
        
        # Send UI context to frontend
        await progress_handler.handle_ui_context_update(workspace_id, ui_context.to_dict())
        
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
            "ui_context": ui_context.to_dict()
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
                individual_analyses.append(result.analysis)
            else:
                # Create error analysis for failed processing
                individual_analyses.append(_create_error_analysis(result))
        
        # Create enhanced compliance report with UI context
        compliance_report = await _create_enhanced_compliance_report(
            individual_analyses, ui_context, performance_monitor, compliance_framework, workspace_id
        )
        
        processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()
        
        # Create comprehensive response with Day 3 enhancements
        analysis_response = AnalysisResponse(
            individual_analyses=individual_analyses,
            compliance_report=compliance_report,
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
                    "performance_monitoring": True
                },
                "ui_context": ui_context.to_dict(),
                "performance_metrics": performance_monitor.get_processing_statistics()
            }
        )
        
        # Notify frontend of completion
        await progress_handler.handle_batch_completed(workspace_id, {
            "documents_analyzed": len(individual_analyses),
            "processing_time": processing_time,
            "compliance_score": compliance_report.compliance_score,
            "german_documents_detected": compliance_report.german_documents_detected,
            "success_rate": len([r for r in processing_results if r.success]) / len(processing_results),
            "performance_grade": _calculate_performance_grade(processing_results),
            "ui_context": ui_context.to_dict()
        })
        
        logger.info(
            "Day 3 enhanced analysis completed successfully",
            workspace_id=workspace_id,
            processing_time=processing_time,
            documents_processed=len(individual_analyses),
            compliance_score=compliance_report.compliance_score,
            performance_grade=_calculate_performance_grade(processing_results)
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

# ADD these helper functions at the end of the file:

def _create_error_analysis(result) -> DocumentAnalysis:
    """Create error analysis for failed processing result"""
    from app.models.compliance import DocumentAnalysis, DocumentLanguage
    
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
    workspace_id: str
) -> ComplianceReport:
    """Create enhanced compliance report with Day 3 intelligence"""
    
    from app.models.compliance import ComplianceReport, DocumentLanguage
    
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
    
    final_compliance_score = min(1.0, base_compliance_score + scenario_bonus + completeness_bonus)
    
    # Enhanced executive summary with UI context
    executive_summary = f"""
Day 3 Enhanced Analysis: {framework.value.upper()} compliance assessment completed.

Scenario Detected: {ui_context.scenario_description}
Industry: {ui_context.industry_detected.value.title()}
German Authority: {ui_context.german_authority.value.upper() if ui_context.german_authority.value != 'unknown' else 'Not specified'}

Processed {len(analyses)} documents with {ui_context.total_documents} total chunks.
Compliance Portfolio Score: {final_compliance_score:.2f}/1.0
German Content: {ui_context.german_content_percentage:.1f}% of documents

Smart Actions Available: {len(ui_context.suggested_actions)} automated recommendations
Priority Risks Identified: {len(ui_context.priority_risks)}
    """.strip()
    
    # Enhanced recommendations based on UI context
    next_steps = [
        action.description for action in ui_context.suggested_actions[:3]
    ]
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