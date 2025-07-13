# app/routers/compliance/__init__.py
"""
Enhanced Compliance Module - Refactored Structure

This module provides enterprise-grade compliance analysis with German authority
support, parallel processing, and real-time progress tracking.

Key Components:
- ComplianceRouter: FastAPI routing layer
- ComplianceOrchestrator: Main business logic coordination
- AuthorityDetectionService: German authority detection and context management
- FileProcessingService: File validation and preprocessing
- PerformanceMonitor: Performance tracking and optimization
- ComplianceResponseBuilder: Response formatting and metadata compilation
"""

from .compliance_router import router as compliance_router
from .compliance_orchestrator import ComplianceOrchestrator
from .authority_detection_service import AuthorityDetectionService, AuthorityContext
from .file_processing_service import FileProcessingService
from .performance_monitoring import PerformanceMonitor
from .compliance_response_builder import ComplianceResponseBuilder

# Version info for debugging
__version__ = "2.0.0"
__refactor_date__ = "2025-07-13"

# Export main router for FastAPI app registration
__all__ = [
    "compliance_router",
    "ComplianceOrchestrator", 
    "AuthorityDetectionService",
    "AuthorityContext",
    "FileProcessingService",
    "PerformanceMonitor",
    "ComplianceResponseBuilder"
]

# Module health check
def get_module_health() -> dict:
    """Get health status of all compliance module components"""
    return {
        "module": "compliance",
        "version": __version__,
        "refactor_date": __refactor_date__,
        "components": {
            "router": "available",
            "orchestrator": "available", 
            "authority_detection": "available",
            "file_processing": "available",
            "performance_monitoring": "available",
            "response_builder": "available"
        },
        "status": "healthy"
    }