# app/services/parallel_processing/__init__.py
"""
Parallel Processing Module for WolfMerge Day 3 Performance Enhancement

This module provides 10x performance improvement through intelligent parallel processing
and UI context detection for German compliance analysis.

Key Components:
- JobQueue: Intelligent job prioritization with German document priority
- BatchProcessor: Async parallel processing with OpenAI rate limiting
- ProgressTracker: Real-time WebSocket progress updates
- UIContextLayer: Smart scenario detection for UI automation
- PerformanceMonitor: Processing metrics and optimization
"""

from .job_queue import JobQueue, DocumentJob, JobPriority
from .batch_processor import BatchProcessor, ProcessingResult
from .progress_tracker import ProgressTracker, ProgressUpdate
from .ui_context_layer import UIContextLayer, UIContext, AuditScenario
from .performance_monitor import PerformanceMonitor, ProcessingMetrics

__all__ = [
    # Core processing
    "JobQueue",
    "DocumentJob", 
    "JobPriority",
    "BatchProcessor",
    "ProcessingResult",
    
    # Progress tracking
    "ProgressTracker",
    "ProgressUpdate",
    
    # UI intelligence
    "UIContextLayer",
    "UIContext",
    "AuditScenario",
    
    # Performance monitoring
    "PerformanceMonitor",
    "ProcessingMetrics"
]

# Version for Day 3 completion tracking
__version__ = "3.0.0"