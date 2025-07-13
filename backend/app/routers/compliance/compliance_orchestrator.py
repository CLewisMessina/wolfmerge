# app/routers/compliance/compliance_orchestrator.py
"""
Compliance Orchestrator Service

Main business logic coordinator that orchestrates the entire compliance analysis workflow.
This service coordinates all other services and manages the analysis process from start to finish.
"""

from typing import List, Optional, Dict, Any
from fastapi import UploadFile, HTTPException
import structlog
import uuid
import time
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.compliance import AnalysisResponse, ComplianceFramework
from app.services.enhanced_compliance_analyzer import EnhancedComplianceAnalyzer
from .authority_detection_service import AuthorityDetectionService, AuthorityContext
from .file_processing_service import FileProcessingService, ProcessedFile
from .performance_monitoring import PerformanceMonitor, ProcessingPhase
from .compliance_response_builder import ComplianceResponseBuilder

logger = structlog.get_logger()

class ComplianceOrchestrator:
    """
    Main orchestrator for compliance analysis workflow.
    
    This service coordinates all aspects of compliance analysis:
    - File processing and validation
    - Authority detection and context management
    - Performance monitoring and optimization
    - Compliance analysis execution
    - Response building and formatting
    
    The orchestrator pattern centralizes workflow logic while keeping
    individual services focused on their specific responsibilities.
    """
    
    def __init__(self, db_session: AsyncSession):
        self.db_session = db_session
        
        # Initialize all service dependencies
        self.file_service = FileProcessingService()
        self.authority_service = AuthorityDetectionService()
        self.performance_monitor = PerformanceMonitor()
        self.response_builder = ComplianceResponseBuilder()
        self.compliance_analyzer = EnhancedComplianceAnalyzer(db_session)
        
        logger.info("Compliance orchestrator initialized with all services")
    
    async def process_compliance_analysis(
        self,
        files: List[UploadFile],
        framework: str = "gdpr",
        workspace_id: str = None,
        user_id: str = None,
        company_location: Optional[str] = None,
        industry_hint: Optional[str] = None,
        client_ip: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> AnalysisResponse:
        """
        Main entry point for compliance analysis processing.
        
        This method orchestrates the entire analysis workflow:
        1. Parameter validation and setup
        2. File processing and validation
        3. Authority detection and context creation
        4. Performance monitoring setup
        5. Compliance analysis execution
        6. Response building and formatting
        """
        
        # Generate unique session ID for tracking
        session_id = str(uuid.uuid4())
        
        try:
            # Validate and convert framework
            compliance_framework = self._validate_framework(framework)
            
            # Start performance monitoring
            await self._initialize_performance_monitoring(session_id, files)
            
            # Phase 1: File Processing and Validation
            processed_files = await self._process_files_phase(session_id, files)
            
            # Phase 2: Authority Detection and Context Creation
            authority_context = await self._authority_detection_phase(
                session_id, processed_files, company_location, industry_hint
            )
            
            # Phase 3: Compliance Analysis Execution
            individual_analyses, compliance_report = await self._compliance_analysis_phase(
                session_id, processed_files, compliance_framework, 
                workspace_id, user_id, authority_context
            )
            
            # Phase 4: Response Building and Finalization
            final_response = await self._response_building_phase(
                session_id, individual_analyses, compliance_report,
                authority_context, processed_files, compliance_framework,
                workspace_id, user_id
            )
            
            # Complete performance monitoring
            performance_metrics = self.performance_monitor.complete_session(session_id)
            
            logger.info(
                "Compliance analysis completed successfully",
                session_id=session_id,
                documents=len(processed_files),
                grade=performance_metrics.grade.value if performance_metrics else "N/A",
                authority=authority_context.detected_authority.value,
                framework=compliance_framework.value
            )
            
            return final_response
            
        except Exception as e:
            # Handle errors and build error response
            await self._handle_analysis_error(session_id, e, workspace_id, user_id)
            raise
    
    def _validate_framework(self, framework: str) -> ComplianceFramework:
        """Validate and convert framework string to enum"""
        try:
            return ComplianceFramework(framework.lower())
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported framework: {framework}. Supported: gdpr, soc2, hipaa, iso27001"
            )
    
    async def _initialize_performance_monitoring(
        self, 
        session_id: str, 
        files: List[UploadFile]
    ) -> None:
        """Initialize performance monitoring for the session"""
        
        # Calculate total size for monitoring
        total_size_mb = 0
        for file in files:
            # Peek at file size without consuming the stream
            content = await file.read()
            total_size_mb += len(content) / (1024 * 1024)
            await file.seek(0)  # Reset stream position
        
        self.performance_monitor.start_session(
            session_id=session_id,
            total_files=len(files),
            total_size_mb=total_size_mb
        )
        
        logger.debug(f"Performance monitoring initialized for session {session_id}")
    
    async def _process_files_phase(
        self, 
        session_id: str, 
        files: List[UploadFile]
    ) -> List[ProcessedFile]:
        """Phase 1: File processing and validation"""
        
        self.performance_monitor.start_phase(session_id, ProcessingPhase.FILE_VALIDATION)
        
        try:
            processed_files = await self.file_service.process_and_validate_files(files)
            
            self.performance_monitor.complete_phase(
                session_id, 
                ProcessingPhase.FILE_VALIDATION,
                success=True,
                items_processed=len(processed_files)
            )
            
            logger.info(
                f"File processing phase completed",
                session_id=session_id,
                files_processed=len(processed_files),
                total_size_mb=sum(f.size for f in processed_files) / (1024 * 1024)
            )
            
            return processed_files
            
        except Exception as e:
            self.performance_monitor.complete_phase(
                session_id,
                ProcessingPhase.FILE_VALIDATION,
                success=False,
                error=str(e)
            )
            logger.error(f"File processing phase failed: {e}")
            raise HTTPException(
                status_code=400,
                detail=f"File processing failed: {str(e)}"
            )
    
    async def _authority_detection_phase(
        self,
        session_id: str,
        processed_files: List[ProcessedFile],
        company_location: Optional[str],
        industry_hint: Optional[str]
    ) -> AuthorityContext:
        """Phase 2: Authority detection and context creation"""
        
        self.performance_monitor.start_phase(session_id, ProcessingPhase.AUTHORITY_DETECTION)
        
        try:
            # Convert ProcessedFile objects to tuples for authority service
            file_tuples = self.file_service.convert_to_tuples(processed_files)
            
            authority_context = await self.authority_service.detect_authority_context(
                file_tuples, company_location, industry_hint
            )
            
            self.performance_monitor.complete_phase(
                session_id,
                ProcessingPhase.AUTHORITY_DETECTION,
                success=authority_context.is_analysis_complete(),
                items_processed=1
            )
            
            logger.info(
                f"Authority detection phase completed",
                session_id=session_id,
                authority=authority_context.detected_authority.value,
                industry=authority_context.detected_industry.value,
                german_content=authority_context.has_german_content()
            )
            
            return authority_context
            
        except Exception as e:
            self.performance_monitor.complete_phase(
                session_id,
                ProcessingPhase.AUTHORITY_DETECTION,
                success=False,
                error=str(e)
            )
            logger.error(f"Authority detection phase failed: {e}")
            
            # Create fallback authority context
            fallback_context = AuthorityContext()
            fallback_context.processing_errors.append(f"Authority detection failed: {str(e)}")
            return fallback_context
    
    async def _compliance_analysis_phase(
        self,
        session_id: str,
        processed_files: List[ProcessedFile],
        framework: ComplianceFramework,
        workspace_id: str,
        user_id: str,
        authority_context: AuthorityContext
    ) -> tuple:
        """Phase 3: Compliance analysis execution"""
        
        self.performance_monitor.start_phase(session_id, ProcessingPhase.COMPLIANCE_ANALYSIS)
        
        try:
            # Convert ProcessedFile objects to tuples for analyzer
            file_tuples = self.file_service.convert_to_tuples(processed_files)
            
            # Create progress callback for real-time updates
            progress_callback = self.performance_monitor.create_progress_callback(session_id)
            
            # Execute compliance analysis
            analysis_response = await self.compliance_analyzer.analyze_documents_for_workspace(
                files=file_tuples,
                workspace_id=workspace_id,
                user_id=user_id,
                framework=framework,
                progress_callback=progress_callback
            )
            
            # Update file processing results
            successful_files = len(analysis_response.individual_analyses)
            failed_files = len(processed_files) - successful_files
            
            self.performance_monitor.update_file_results(
                session_id, successful_files, failed_files
            )
            
            self.performance_monitor.complete_phase(
                session_id,
                ProcessingPhase.COMPLIANCE_ANALYSIS,
                success=True,
                items_processed=successful_files
            )
            
            logger.info(
                f"Compliance analysis phase completed",
                session_id=session_id,
                successful_files=successful_files,
                failed_files=failed_files,
                compliance_score=analysis_response.compliance_report.compliance_score
            )
            
            return analysis_response.individual_analyses, analysis_response.compliance_report
            
        except Exception as e:
            self.performance_monitor.complete_phase(
                session_id,
                ProcessingPhase.COMPLIANCE_ANALYSIS,
                success=False,
                error=str(e)
            )
            logger.error(f"Compliance analysis phase failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Compliance analysis failed: {str(e)}"
            )
    
    async def _response_building_phase(
        self,
        session_id: str,
        individual_analyses: List,
        compliance_report,
        authority_context: AuthorityContext,
        processed_files: List[ProcessedFile],
        framework: ComplianceFramework,
        workspace_id: str,
        user_id: str
    ) -> AnalysisResponse:
        """Phase 4: Response building and finalization"""
        
        self.performance_monitor.start_phase(session_id, ProcessingPhase.RESPONSE_BUILDING)
        
        try:
            # Get current performance metrics
            performance_metrics = self.performance_monitor.get_session_metrics(session_id)
            
            # Build comprehensive response
            final_response = self.response_builder.build_analysis_response(
                individual_analyses=individual_analyses,
                compliance_report=compliance_report,
                authority_context=authority_context,
                performance_metrics=performance_metrics,
                processed_files=processed_files,
                session_id=session_id,
                workspace_id=workspace_id,
                user_id=user_id,
                framework=framework
            )
            
            self.performance_monitor.complete_phase(
                session_id,
                ProcessingPhase.RESPONSE_BUILDING,
                success=True,
                items_processed=1
            )
            
            logger.info(
                f"Response building phase completed",
                session_id=session_id,
                response_size=len(str(final_response))
            )
            
            return final_response
            
        except Exception as e:
            self.performance_monitor.complete_phase(
                session_id,
                ProcessingPhase.RESPONSE_BUILDING,
                success=False,
                error=str(e)
            )
            logger.error(f"Response building phase failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Response building failed: {str(e)}"
            )
    
    async def _handle_analysis_error(
        self,
        session_id: str,
        error: Exception,
        workspace_id: Optional[str],
        user_id: Optional[str]
    ) -> None:
        """Handle errors during analysis and ensure proper cleanup"""
        
        try:
            # Complete performance monitoring with error
            self.performance_monitor.complete_session(session_id)
            
            # Log error details
            logger.error(
                "Compliance analysis failed",
                session_id=session_id,
                workspace_id=workspace_id,
                user_id=user_id,
                error=str(error),
                exc_info=True
            )
            
            # Build error response for WebSocket clients
            error_response = self.response_builder.build_error_response(
                session_id=session_id,
                error_message=str(error),
                error_type=self._classify_error_type(error),
                workspace_id=workspace_id,
                user_id=user_id
            )
            
            # Send error notification via WebSocket if possible
            try:
                from app.services.websocket.progress_handler import progress_handler
                await progress_handler.send_progress_update(session_id, {
                    "type": "analysis_error",
                    "error": error_response
                })
            except Exception as ws_error:
                logger.warning(f"Failed to send error notification via WebSocket: {ws_error}")
            
        except Exception as cleanup_error:
            logger.error(f"Error cleanup failed: {cleanup_error}")
    
    def _classify_error_type(self, error: Exception) -> str:
        """Classify error type for better error handling"""
        
        error_str = str(error).lower()
        
        if "file" in error_str and ("size" in error_str or "format" in error_str):
            return "file_validation_error"
        elif "authority" in error_str or "detection" in error_str:
            return "authority_detection_error"
        elif "timeout" in error_str or "time" in error_str:
            return "processing_timeout"
        elif "docling" in error_str:
            return "docling_error"
        elif "memory" in error_str or "resource" in error_str:
            return "resource_error"
        else:
            return "processing_error"
    
    async def get_session_status(self, session_id: str) -> Dict[str, Any]:
        """Get current status of an active analysis session"""
        
        performance_metrics = self.performance_monitor.get_session_metrics(session_id)
        
        if not performance_metrics:
            return {"error": "Session not found", "session_id": session_id}
        
        return {
            "session_id": session_id,
            "status": "active",
            "progress": self._calculate_session_progress(performance_metrics),
            "performance": {
                "current_duration": time.time() - performance_metrics.start_time,
                "files_processed": performance_metrics.successful_files,
                "files_remaining": performance_metrics.total_files - performance_metrics.successful_files,
                "current_phase": self._get_current_phase(performance_metrics)
            }
        }
    
    def _calculate_session_progress(self, performance_metrics) -> float:
        """Calculate session progress percentage"""
        
        total_phases = len(ProcessingPhase)
        completed_phases = sum(1 for p in performance_metrics.phases.values() if p.end_time)
        
        return (completed_phases / total_phases) * 100
    
    def _get_current_phase(self, performance_metrics) -> str:
        """Get current processing phase"""
        
        for phase, metrics in performance_metrics.phases.items():
            if not metrics.end_time:  # Phase started but not completed
                return phase.value
        
        return "completed"
    
    async def get_performance_summary(self) -> Dict[str, Any]:
        """Get overall performance summary across all sessions"""
        
        return self.performance_monitor.get_performance_summary()
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check of all orchestrator components"""
        
        health_status = {
            "orchestrator": "healthy",
            "timestamp": time.time(),
            "components": {}
        }
        
        # Check each service component
        try:
            # File service health
            health_status["components"]["file_service"] = {
                "status": "healthy",
                "max_file_size": self.file_service.max_file_size,
                "max_batch_size": self.file_service.max_batch_size
            }
        except Exception as e:
            health_status["components"]["file_service"] = {
                "status": "unhealthy",
                "error": str(e)
            }
        
        try:
            # Authority service health
            health_status["components"]["authority_service"] = {
                "status": "healthy",
                "big4_engine_available": self.authority_service.big4_endpoints is not None
            }
        except Exception as e:
            health_status["components"]["authority_service"] = {
                "status": "unhealthy", 
                "error": str(e)
            }
        
        try:
            # Performance monitor health
            active_sessions = len(self.performance_monitor.active_sessions)
            health_status["components"]["performance_monitor"] = {
                "status": "healthy",
                "active_sessions": active_sessions,
                "historical_sessions": len(self.performance_monitor.historical_metrics)
            }
        except Exception as e:
            health_status["components"]["performance_monitor"] = {
                "status": "unhealthy",
                "error": str(e)
            }
        
        try:
            # Response builder health
            health_status["components"]["response_builder"] = self.response_builder.build_health_check_response()
        except Exception as e:
            health_status["components"]["response_builder"] = {
                "status": "unhealthy",
                "error": str(e)
            }
        
        try:
            # Compliance analyzer health (basic check)
            health_status["components"]["compliance_analyzer"] = {
                "status": "healthy",
                "database_connected": self.db_session is not None
            }
        except Exception as e:
            health_status["components"]["compliance_analyzer"] = {
                "status": "unhealthy",
                "error": str(e)
            }
        
        # Overall health determination
        unhealthy_components = [
            name for name, status in health_status["components"].items()
            if status.get("status") == "unhealthy"
        ]
        
        if unhealthy_components:
            health_status["orchestrator"] = "degraded"
            health_status["unhealthy_components"] = unhealthy_components
        
        return health_status