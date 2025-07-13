# app/routers/compliance/compliance_response_builder.py
"""
Compliance Response Builder Service

Handles formatting of analysis results, metadata compilation, authority context
serialization, and building comprehensive API responses for compliance analysis.
"""

from typing import Dict, Any, List, Optional
import structlog
import time
from datetime import datetime, timezone

from app.models.compliance import (
    AnalysisResponse, DocumentAnalysis, ComplianceReport, ComplianceFramework
)
from .authority_detection_service import AuthorityContext
from .performance_monitoring import PerformanceMetrics
from .file_processing_service import ProcessedFile

logger = structlog.get_logger()

class ComplianceResponseBuilder:
    """
    Service for building comprehensive compliance analysis responses.
    
    This service handles:
    - Formatting analysis results for API consumption
    - Compiling metadata from all processing stages  
    - Serializing authority context and performance metrics
    - Building user-friendly summaries and recommendations
    - Creating audit trail information
    """
    
    def __init__(self):
        self.response_version = "2.0.0"
        self.build_timestamp = datetime.now(timezone.utc).isoformat()
    
    def build_analysis_response(
        self,
        individual_analyses: List[DocumentAnalysis],
        compliance_report: ComplianceReport,
        authority_context: AuthorityContext,
        performance_metrics: PerformanceMetrics,
        processed_files: List[ProcessedFile],
        session_id: str,
        workspace_id: str,
        user_id: str,
        framework: ComplianceFramework
    ) -> AnalysisResponse:
        """
        Build comprehensive analysis response with all metadata.
        
        This is the main entry point for creating the final API response
        that includes all analysis results, metadata, and recommendations.
        """
        
        # Build processing metadata
        processing_metadata = self._build_processing_metadata(
            authority_context, performance_metrics, processed_files, 
            session_id, workspace_id, user_id, framework
        )
        
        # Enhance compliance report with authority insights
        enhanced_report = self._enhance_compliance_report(
            compliance_report, authority_context, performance_metrics
        )
        
        # Create the response object
        response = AnalysisResponse(
            individual_analyses=individual_analyses,
            compliance_report=enhanced_report,
            processing_metadata=processing_metadata
        )
        
        logger.info(
            "Analysis response built successfully",
            session_id=session_id,
            document_count=len(individual_analyses),
            compliance_score=enhanced_report.compliance_score,
            authority=authority_context.detected_authority.value,
            performance_grade=performance_metrics.grade.value if performance_metrics.grade else "N/A"
        )
        
        return response
    
    def _build_processing_metadata(
        self,
        authority_context: AuthorityContext,
        performance_metrics: PerformanceMetrics,
        processed_files: List[ProcessedFile],
        session_id: str,
        workspace_id: str,
        user_id: str,
        framework: ComplianceFramework
    ) -> Dict[str, Any]:
        """Build comprehensive processing metadata"""
        
        return {
            # Session information
            "session_info": {
                "session_id": session_id,
                "workspace_id": workspace_id,
                "user_id": user_id,
                "framework": framework.value,
                "timestamp": self.build_timestamp,
                "response_version": self.response_version
            },
            
            # File processing metadata
            "file_processing": self._build_file_metadata(processed_files),
            
            # Authority detection results
            "authority_detection": authority_context.to_api_response(),
            
            # Performance metrics
            "performance": self._build_performance_metadata(performance_metrics),
            
            # Processing summary
            "processing_summary": self._build_processing_summary(
                authority_context, performance_metrics, processed_files
            ),
            
            # System information
            "system_info": {
                "processing_node": "primary",
                "api_version": "v2",
                "docling_enabled": any(f.metadata.docling_recommended for f in processed_files),
                "parallel_processing": performance_metrics.total_files > 1,
                "german_optimization": authority_context.has_german_content()
            }
        }
    
    def _build_file_metadata(self, processed_files: List[ProcessedFile]) -> Dict[str, Any]:
        """Build file processing metadata"""
        
        if not processed_files:
            return {"files_processed": 0}
        
        total_size = sum(f.size for f in processed_files)
        
        # File type distribution
        mime_types = {}
        for f in processed_files:
            mime_type = f.metadata.mime_type
            mime_types[mime_type] = mime_types.get(mime_type, 0) + 1
        
        # Security analysis
        security_flags = []
        for f in processed_files:
            security_flags.extend(f.metadata.security_flags)
        
        # Docling usage
        docling_files = sum(1 for f in processed_files if f.metadata.docling_recommended)
        
        return {
            "files_processed": len(processed_files),
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "file_types": mime_types,
            "docling_processing": {
                "files_using_docling": docling_files,
                "percentage": round((docling_files / len(processed_files)) * 100, 1)
            },
            "security_analysis": {
                "total_security_flags": len(security_flags),
                "unique_security_issues": len(set(security_flags)),
                "files_with_security_flags": sum(1 for f in processed_files if f.metadata.security_flags)
            },
            "complexity_analysis": {
                "average_complexity": round(
                    sum(f.metadata.processing_complexity for f in processed_files) / len(processed_files), 2
                ),
                "total_estimated_time": round(
                    sum(f.metadata.estimated_processing_time for f in processed_files), 2
                )
            }
        }
    
    def _build_performance_metadata(self, performance_metrics: PerformanceMetrics) -> Dict[str, Any]:
        """Build performance monitoring metadata"""
        
        if not performance_metrics:
            return {"performance_monitoring": "disabled"}
        
        # Phase timing breakdown
        phase_timings = {}
        for phase, metrics in performance_metrics.phases.items():
            if metrics.duration:
                phase_timings[phase.value] = {
                    "duration_seconds": round(metrics.duration, 3),
                    "success": metrics.success,
                    "items_processed": metrics.items_processed
                }
                if metrics.error_message:
                    phase_timings[phase.value]["error"] = metrics.error_message
        
        return {
            "overall_performance": {
                "grade": performance_metrics.grade.value if performance_metrics.grade else "N/A",
                "total_duration": round(performance_metrics.total_duration or 0, 2),
                "success_rate": round(performance_metrics.success_rate or 0, 3),
                "throughput_files_per_second": round(performance_metrics.throughput_files_per_second or 0, 2),
                "throughput_mb_per_second": round(performance_metrics.throughput_mb_per_second or 0, 2)
            },
            "phase_breakdown": phase_timings,
            "optimization": {
                "bottlenecks_identified": performance_metrics.bottlenecks,
                "recommendations": performance_metrics.recommendations[:5]  # Top 5 recommendations
            },
            "targets": {
                "target_time_per_file": "3.0 seconds",
                "target_success_rate": "95%",
                "current_vs_target": self._calculate_target_comparison(performance_metrics)
            }
        }
    
    def _calculate_target_comparison(self, performance_metrics: PerformanceMetrics) -> Dict[str, str]:
        """Compare current performance to targets"""
        
        if not performance_metrics.total_duration or not performance_metrics.total_files:
            return {"status": "insufficient_data"}
        
        avg_time_per_file = performance_metrics.total_duration / performance_metrics.total_files
        success_rate = performance_metrics.success_rate or 0
        
        time_status = "✅ Under target" if avg_time_per_file <= 3.0 else f"⚠️ {avg_time_per_file:.1f}s (target: 3.0s)"
        success_status = "✅ Above target" if success_rate >= 0.95 else f"⚠️ {success_rate:.1%} (target: 95%)"
        
        return {
            "processing_time": time_status,
            "success_rate": success_status,
            "overall_status": "✅ Meeting targets" if avg_time_per_file <= 3.0 and success_rate >= 0.95 else "⚠️ Below targets"
        }
    
    def _build_processing_summary(
        self,
        authority_context: AuthorityContext,
        performance_metrics: PerformanceMetrics,
        processed_files: List[ProcessedFile]
    ) -> Dict[str, Any]:
        """Build high-level processing summary"""
        
        return {
            "analysis_scope": {
                "documents_processed": len(processed_files),
                "german_content_detected": authority_context.has_german_content(),
                "authority_identified": authority_context.has_valid_authority_data(),
                "big4_analysis_performed": authority_context.big4_engine_available and authority_context.authority_analysis is not None
            },
            "key_findings": {
                "primary_authority": authority_context.detected_authority.value,
                "industry_classification": authority_context.detected_industry.value,
                "risk_level": authority_context.get_risk_level(),
                "audit_readiness": f"{authority_context.audit_readiness_score:.1%}" if authority_context.audit_readiness_score else "N/A"
            },
            "processing_efficiency": {
                "performance_grade": performance_metrics.grade.value if performance_metrics.grade else "N/A",
                "processing_speed": f"{performance_metrics.throughput_files_per_second:.1f} files/sec" if performance_metrics.throughput_files_per_second else "N/A",
                "optimization_needed": len(performance_metrics.bottlenecks) > 0
            },
            "next_steps": self._generate_next_steps(authority_context, performance_metrics)
        }
    
    def _enhance_compliance_report(
        self,
        base_report: ComplianceReport,
        authority_context: AuthorityContext,
        performance_metrics: PerformanceMetrics
    ) -> ComplianceReport:
        """Enhance compliance report with authority and performance insights"""
        
        # Add German-specific recommendations if relevant
        german_recommendations = []
        if authority_context.has_german_content():
            german_recommendations.extend(authority_context.authority_guidance)
            
            # Add authority-specific recommendations
            if authority_context.detected_authority.value != "unknown":
                german_recommendations.append(
                    f"Prepare compliance documentation for {authority_context.detected_authority.value.upper()} requirements"
                )
        
        # Add performance-based recommendations
        performance_recommendations = []
        if performance_metrics.grade and performance_metrics.grade.value in ["D", "F"]:
            performance_recommendations.append("Consider optimizing document processing workflow")
        
        # Update the report
        enhanced_report = ComplianceReport(
            framework=base_report.framework,
            executive_summary=base_report.executive_summary,
            executive_summary_de=self._generate_german_summary(base_report, authority_context),
            compliance_score=base_report.compliance_score,
            documents_analyzed=base_report.documents_analyzed,
            german_documents_detected=authority_context.has_german_content(),
            priority_gaps=base_report.priority_gaps,
            compliance_strengths=base_report.compliance_strengths,
            next_steps=base_report.next_steps + performance_recommendations,
            german_specific_recommendations=german_recommendations
        )
        
        return enhanced_report
    
    def _generate_german_summary(
        self,
        base_report: ComplianceReport,
        authority_context: AuthorityContext
    ) -> Optional[str]:
        """Generate German executive summary if German content was detected"""
        
        if not authority_context.has_german_content():
            return None
        
        # Basic German summary template
        german_summary = f"""
        DSGVO-Compliance Zusammenfassung:
        
        Compliance-Score: {base_report.compliance_score:.1%}
        Analysierte Dokumente: {base_report.documents_analyzed}
        Erkannte Aufsichtsbehörde: {authority_context.detected_authority.value.upper()}
        Branche: {authority_context.detected_industry.value}
        
        Wichtigste Erkenntnisse: Deutsche Datenschutzinhalte wurden erkannt und mit 
        branchenspezifischen Compliance-Anforderungen abgeglichen.
        """
        
        return german_summary.strip()
    
    def _generate_next_steps(
        self,
        authority_context: AuthorityContext,
        performance_metrics: PerformanceMetrics
    ) -> List[str]:
        """Generate actionable next steps based on analysis results"""
        
        next_steps = []
        
        # Authority-specific next steps
        if authority_context.has_valid_authority_data():
            next_steps.append(f"Review {authority_context.detected_authority.value.upper()} specific requirements")
            
            if authority_context.penalty_risk_level in ["high", "critical"]:
                next_steps.append("Conduct immediate compliance audit")
            
            if authority_context.audit_readiness_score < 0.8:
                next_steps.append("Improve audit readiness documentation")
        
        # Performance-based next steps
        if performance_metrics.grade and performance_metrics.grade.value in ["D", "F"]:
            next_steps.append("Optimize document processing performance")
        
        if performance_metrics.bottlenecks:
            next_steps.append("Address identified processing bottlenecks")
        
        # German content specific steps
        if authority_context.has_german_content():
            next_steps.append("Ensure DSGVO compliance documentation is current")
            next_steps.append("Consider German data localization requirements")
        
        # Default recommendations if no specific issues found
        if not next_steps:
            next_steps.extend([
                "Continue regular compliance monitoring",
                "Keep documentation updated with regulatory changes",
                "Schedule periodic compliance reviews"
            ])
        
        return next_steps[:8]  # Limit to 8 actionable items
    
    def build_error_response(
        self,
        session_id: str,
        error_message: str,
        error_type: str = "processing_error",
        workspace_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Build standardized error response"""
        
        return {
            "error": True,
            "error_type": error_type,
            "error_message": error_message,
            "session_id": session_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metadata": {
                "workspace_id": workspace_id,
                "user_id": user_id,
                "response_version": self.response_version
            },
            "suggestions": self._get_error_suggestions(error_type)
        }
    
    def _get_error_suggestions(self, error_type: str) -> List[str]:
        """Get helpful suggestions based on error type"""
        
        suggestions_map = {
            "file_validation_error": [
                "Check file format is supported (.pdf, .docx, .txt, etc.)",
                "Ensure file size is under the maximum limit",
                "Verify file is not corrupted"
            ],
            "authority_detection_error": [
                "Ensure documents contain German compliance content",
                "Check if company location is specified correctly",
                "Verify industry hint is accurate"
            ],
            "processing_timeout": [
                "Try processing fewer files at once",
                "Consider splitting large documents",
                "Check system performance and retry"
            ],
            "docling_error": [
                "Try processing without Docling optimization",
                "Ensure document format is supported",
                "Check if document is password protected"
            ]
        }
        
        return suggestions_map.get(error_type, [
            "Check input parameters and retry",
            "Contact support if problem persists",
            "Review API documentation for requirements"
        ])
    
    def build_health_check_response(self) -> Dict[str, Any]:
        """Build health check response for monitoring"""
        
        return {
            "service": "compliance_response_builder",
            "status": "healthy",
            "version": self.response_version,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "capabilities": [
                "analysis_response_building",
                "metadata_compilation",
                "authority_context_serialization",
                "performance_metrics_integration",
                "german_content_support",
                "error_response_handling"
            ]
        }