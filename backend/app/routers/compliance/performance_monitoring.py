# app/routers/compliance/performance_monitoring.py
"""
Performance Monitoring Service

Handles performance tracking, optimization recommendations, WebSocket progress updates,
and the A-F grading system for compliance analysis performance.
"""

from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import time
import asyncio
import structlog
from datetime import datetime, timezone

from app.services.websocket.progress_handler import progress_handler

logger = structlog.get_logger()

class PerformanceGrade(str, Enum):
    """Performance grades based on processing time and success rate"""
    A = "A"  # <= 3.0s, >= 95% success
    B = "B"  # <= 5.0s, >= 90% success  
    C = "C"  # <= 8.0s, >= 80% success
    D = "D"  # <= 12.0s, >= 70% success
    F = "F"  # > 12.0s or < 70% success

class ProcessingPhase(str, Enum):
    """Different phases of document processing"""
    INITIALIZATION = "initialization"
    FILE_VALIDATION = "file_validation"
    AUTHORITY_DETECTION = "authority_detection"
    DOCUMENT_PROCESSING = "document_processing"
    COMPLIANCE_ANALYSIS = "compliance_analysis"
    RESPONSE_BUILDING = "response_building"
    FINALIZATION = "finalization"

@dataclass
class PhaseMetrics:
    """Metrics for a specific processing phase"""
    phase: ProcessingPhase
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    success: bool = True
    error_message: Optional[str] = None
    items_processed: int = 0
    
    def complete(self, success: bool = True, error: Optional[str] = None) -> None:
        """Mark phase as complete and calculate duration"""
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        self.success = success
        self.error_message = error

@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics for an analysis session"""
    session_id: str
    start_time: float
    end_time: Optional[float] = None
    total_duration: Optional[float] = None
    
    # File metrics
    total_files: int = 0
    successful_files: int = 0
    failed_files: int = 0
    total_size_mb: float = 0.0
    
    # Phase tracking
    phases: Dict[ProcessingPhase, PhaseMetrics] = field(default_factory=dict)
    
    # Performance indicators
    throughput_files_per_second: Optional[float] = None
    throughput_mb_per_second: Optional[float] = None
    success_rate: Optional[float] = None
    grade: Optional[PerformanceGrade] = None
    
    # Optimization recommendations
    bottlenecks: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    def complete_session(self) -> None:
        """Mark session as complete and calculate final metrics"""
        self.end_time = time.time()
        self.total_duration = self.end_time - self.start_time
        
        if self.total_duration > 0:
            self.throughput_files_per_second = self.total_files / self.total_duration
            self.throughput_mb_per_second = self.total_size_mb / self.total_duration
        
        if self.total_files > 0:
            self.success_rate = self.successful_files / self.total_files
        
        self.grade = self._calculate_grade()
        self._identify_bottlenecks()
        self._generate_recommendations()
    
    def _calculate_grade(self) -> PerformanceGrade:
        """Calculate performance grade based on time and success rate"""
        if not self.total_duration or not self.success_rate:
            return PerformanceGrade.F
        
        avg_time_per_file = self.total_duration / max(1, self.total_files)
        
        if avg_time_per_file <= 3.0 and self.success_rate >= 0.95:
            return PerformanceGrade.A
        elif avg_time_per_file <= 5.0 and self.success_rate >= 0.90:
            return PerformanceGrade.B
        elif avg_time_per_file <= 8.0 and self.success_rate >= 0.80:
            return PerformanceGrade.C
        elif avg_time_per_file <= 12.0 and self.success_rate >= 0.70:
            return PerformanceGrade.D
        else:
            return PerformanceGrade.F
    
    def _identify_bottlenecks(self) -> None:
        """Identify performance bottlenecks based on phase timing"""
        if not self.phases:
            return
        
        # Find slowest phases
        phase_durations = {
            phase: metrics.duration or 0
            for phase, metrics in self.phases.items()
            if metrics.duration
        }
        
        if not phase_durations:
            return
        
        total_phase_time = sum(phase_durations.values())
        
        for phase, duration in phase_durations.items():
            percentage = (duration / total_phase_time) * 100
            
            # Flag phases taking more than 30% of total time
            if percentage > 30:
                self.bottlenecks.append(f"{phase.value}_slow_{percentage:.1f}%")
            
            # Flag failed phases
            if not self.phases[phase].success:
                self.bottlenecks.append(f"{phase.value}_failed")
    
    def _generate_recommendations(self) -> None:
        """Generate optimization recommendations based on performance data"""
        if not self.total_duration:
            return
        
        avg_time_per_file = self.total_duration / max(1, self.total_files)
        
        # Time-based recommendations
        if avg_time_per_file > 8.0:
            self.recommendations.append("Consider enabling parallel processing")
            self.recommendations.append("Optimize document chunking strategy")
        
        if avg_time_per_file > 5.0:
            self.recommendations.append("Enable Docling optimization mode")
        
        # Success rate recommendations  
        if self.success_rate and self.success_rate < 0.9:
            self.recommendations.append("Review file validation settings")
            self.recommendations.append("Add error recovery mechanisms")
        
        # Bottleneck-specific recommendations
        for bottleneck in self.bottlenecks:
            if "document_processing_slow" in bottleneck:
                self.recommendations.append("Optimize Docling processing settings")
            elif "compliance_analysis_slow" in bottleneck:
                self.recommendations.append("Consider AI model optimization")
            elif "authority_detection_slow" in bottleneck:
                self.recommendations.append("Cache authority detection results")

class PerformanceMonitor:
    """
    Service for monitoring and optimizing compliance analysis performance.
    
    Tracks processing times, success rates, bottlenecks, and provides
    optimization recommendations. Integrates with WebSocket for real-time
    progress updates.
    """
    
    def __init__(self):
        self.active_sessions: Dict[str, PerformanceMetrics] = {}
        self.historical_metrics: List[PerformanceMetrics] = []
        self.max_historical_sessions = 100
        
        # Performance targets for optimization
        self.targets = {
            "max_time_per_file": 3.0,  # seconds
            "min_success_rate": 0.95,
            "max_total_time": 30.0  # seconds for typical batch
        }
    
    def start_session(
        self, 
        session_id: str, 
        total_files: int, 
        total_size_mb: float
    ) -> PerformanceMetrics:
        """Start monitoring a new analysis session"""
        
        metrics = PerformanceMetrics(
            session_id=session_id,
            start_time=time.time(),
            total_files=total_files,
            total_size_mb=total_size_mb
        )
        
        self.active_sessions[session_id] = metrics
        
        logger.info(
            "Performance monitoring started",
            session_id=session_id,
            total_files=total_files,
            total_size_mb=total_size_mb
        )
        
        return metrics
    
    def start_phase(self, session_id: str, phase: ProcessingPhase) -> None:
        """Start monitoring a specific processing phase"""
        
        if session_id not in self.active_sessions:
            logger.warning(f"Unknown session {session_id} for phase {phase}")
            return
        
        phase_metrics = PhaseMetrics(
            phase=phase,
            start_time=time.time()
        )
        
        self.active_sessions[session_id].phases[phase] = phase_metrics
        
        logger.debug(f"Phase {phase.value} started for session {session_id}")
    
    def complete_phase(
        self, 
        session_id: str, 
        phase: ProcessingPhase, 
        success: bool = True, 
        error: Optional[str] = None,
        items_processed: int = 0
    ) -> None:
        """Mark a processing phase as complete"""
        
        if session_id not in self.active_sessions:
            logger.warning(f"Unknown session {session_id} for phase completion {phase}")
            return
        
        session = self.active_sessions[session_id]
        
        if phase not in session.phases:
            logger.warning(f"Phase {phase} not started for session {session_id}")
            return
        
        session.phases[phase].complete(success, error)
        session.phases[phase].items_processed = items_processed
        
        # Send WebSocket progress update
        asyncio.create_task(self._send_progress_update(session_id, phase, success))
        
        logger.debug(
            f"Phase {phase.value} completed",
            session_id=session_id,
            duration=session.phases[phase].duration,
            success=success
        )
    
    def update_file_results(
        self, 
        session_id: str, 
        successful_files: int, 
        failed_files: int
    ) -> None:
        """Update file processing results"""
        
        if session_id not in self.active_sessions:
            return
        
        session = self.active_sessions[session_id]
        session.successful_files = successful_files
        session.failed_files = failed_files
    
    def complete_session(self, session_id: str) -> PerformanceMetrics:
        """Complete monitoring session and generate final metrics"""
        
        if session_id not in self.active_sessions:
            logger.warning(f"Unknown session {session_id} for completion")
            return None
        
        session = self.active_sessions[session_id]
        session.complete_session()
        
        # Move to historical metrics
        self.historical_metrics.append(session)
        if len(self.historical_metrics) > self.max_historical_sessions:
            self.historical_metrics.pop(0)
        
        # Remove from active sessions
        del self.active_sessions[session_id]
        
        # Send final WebSocket update
        asyncio.create_task(self._send_final_update(session_id, session))
        
        logger.info(
            "Performance monitoring completed",
            session_id=session_id,
            total_duration=session.total_duration,
            grade=session.grade.value if session.grade else "N/A",
            success_rate=session.success_rate,
            throughput_fps=session.throughput_files_per_second
        )
        
        return session
    
    async def _send_progress_update(
        self, 
        session_id: str, 
        phase: ProcessingPhase, 
        success: bool
    ) -> None:
        """Send WebSocket progress update"""
        
        try:
            session = self.active_sessions.get(session_id)
            if not session:
                return
            
            # Calculate overall progress based on completed phases
            total_phases = len(ProcessingPhase)
            completed_phases = sum(1 for p in session.phases.values() if p.end_time)
            progress_percentage = (completed_phases / total_phases) * 100
            
            update_data = {
                "type": "phase_complete",
                "session_id": session_id,
                "phase": phase.value,
                "success": success,
                "progress_percentage": progress_percentage,
                "current_time": time.time() - session.start_time
            }
            
            await progress_handler.send_progress_update(session_id, update_data)
            
        except Exception as e:
            logger.warning(f"Failed to send progress update: {e}")
    
    async def _send_final_update(self, session_id: str, session: PerformanceMetrics) -> None:
        """Send final WebSocket update with complete metrics"""
        
        try:
            update_data = {
                "type": "analysis_complete",
                "session_id": session_id,
                "performance_metrics": {
                    "total_duration": session.total_duration,
                    "grade": session.grade.value if session.grade else "F",
                    "success_rate": session.success_rate,
                    "throughput_fps": session.throughput_files_per_second,
                    "bottlenecks": session.bottlenecks,
                    "recommendations": session.recommendations
                }
            }
            
            await progress_handler.send_progress_update(session_id, update_data)
            
        except Exception as e:
            logger.warning(f"Failed to send final update: {e}")
    
    def get_session_metrics(self, session_id: str) -> Optional[PerformanceMetrics]:
        """Get current metrics for an active session"""
        return self.active_sessions.get(session_id)
    
    def get_historical_metrics(self, limit: int = 10) -> List[PerformanceMetrics]:
        """Get recent historical performance metrics"""
        return self.historical_metrics[-limit:]
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get overall performance summary across all sessions"""
        
        if not self.historical_metrics:
            return {
                "total_sessions": 0,
                "average_grade": "N/A",
                "average_duration": 0.0,
                "average_success_rate": 0.0
            }
        
        total_sessions = len(self.historical_metrics)
        
        # Calculate averages
        grades = [s.grade for s in self.historical_metrics if s.grade]
        grade_values = {"A": 4, "B": 3, "C": 2, "D": 1, "F": 0}
        avg_grade_value = sum(grade_values.get(g.value, 0) for g in grades) / len(grades) if grades else 0
        avg_grade = next((k for k, v in grade_values.items() if v == round(avg_grade_value)), "F")
        
        avg_duration = sum(s.total_duration for s in self.historical_metrics if s.total_duration) / total_sessions
        avg_success_rate = sum(s.success_rate for s in self.historical_metrics if s.success_rate) / total_sessions
        
        # Common bottlenecks and recommendations
        all_bottlenecks = [b for s in self.historical_metrics for b in s.bottlenecks]
        all_recommendations = [r for s in self.historical_metrics for r in s.recommendations]
        
        from collections import Counter
        common_bottlenecks = Counter(all_bottlenecks).most_common(5)
        common_recommendations = Counter(all_recommendations).most_common(5)
        
        return {
            "total_sessions": total_sessions,
            "average_grade": avg_grade,
            "average_duration": avg_duration,
            "average_success_rate": avg_success_rate,
            "performance_targets": self.targets,
            "common_bottlenecks": [{"issue": b, "frequency": f} for b, f in common_bottlenecks],
            "common_recommendations": [{"recommendation": r, "frequency": f} for r, f in common_recommendations]
        }
    
    def create_progress_callback(self, session_id: str) -> Callable:
        """Create a progress callback function for use with other services"""
        
        async def progress_callback(progress_data):
            """
            Flexible progress callback that handles both dictionary and parameter formats
            
            Supports two calling patterns:
            1. Dictionary format: progress_callback({"type": "analysis_started", "document_count": 5})
            2. Parameter format: progress_callback(current=1, total=5, phase="processing")
            """
            try:
                # Handle dictionary format (from EnhancedComplianceAnalyzer)
                if isinstance(progress_data, dict):
                    # Extract relevant information from dictionary
                    progress_type = progress_data.get("type", "progress_update")
                    current = progress_data.get("current", 0)
                    total = progress_data.get("total", progress_data.get("document_count", 0))
                    phase = progress_data.get("phase", progress_type)
                    
                    # Convert specific progress types to meaningful phases
                    if progress_type == "analysis_started":
                        phase = "initialization"
                        current = 0
                        total = progress_data.get("document_count", 1)
                    elif progress_type == "document_processing":
                        phase = "document_analysis"
                        current = progress_data.get("document_index", 0)
                        total = progress_data.get("total_documents", 1)
                    elif progress_type == "document_completed":
                        phase = "document_analysis"
                        current = progress_data.get("document_index", 1)
                        total = progress_data.get("total_documents", 1)
                    elif progress_type == "generating_report":
                        phase = "report_generation"
                        current = progress_data.get("documents_processed", 0)
                        total = progress_data.get("documents_processed", 1)
                    elif progress_type == "analysis_completed":
                        phase = "completion"
                        current = progress_data.get("total_documents", 1)
                        total = progress_data.get("total_documents", 1)
                    elif progress_type == "analysis_failed":
                        phase = "error"
                        current = 0
                        total = 1
                    
                    # Additional context from dictionary
                    document_name = progress_data.get("document_name", "")
                    status = progress_data.get("status", "processing")
                    error_message = progress_data.get("error", "")
                    
                else:
                    # Handle parameter format (legacy support)
                    # Assume positional arguments: current, total, phase
                    if hasattr(progress_data, '__iter__') and not isinstance(progress_data, str):
                        # Multiple arguments passed as tuple/list
                        args = list(progress_data) if not isinstance(progress_data, list) else progress_data
                        current = args[0] if len(args) > 0 else 0
                        total = args[1] if len(args) > 1 else 1
                        phase = args[2] if len(args) > 2 else "processing"
                    else:
                        # Single argument - treat as current progress
                        current = progress_data if isinstance(progress_data, (int, float)) else 0
                        total = 1
                        phase = "processing"
                    
                    # Default values for parameter format
                    document_name = ""
                    status = "processing"
                    error_message = ""
                
                # Calculate percentage
                percentage = (current / total) * 100 if total > 0 else 0
                
                # Create standardized update data
                update_data = {
                    "type": "progress_update",
                    "session_id": session_id,
                    "current": current,
                    "total": total,
                    "percentage": round(percentage, 1),
                    "phase": phase,
                    "timestamp": time.time()
                }
                
                # Add optional fields if available
                if document_name:
                    update_data["document_name"] = document_name
                if status:
                    update_data["status"] = status
                if error_message:
                    update_data["error"] = error_message
                
                # Send progress update via WebSocket
                try:
                    await progress_handler.send_progress_update(session_id, update_data)
                except NameError:
                    # progress_handler not available - log instead
                    logger.info(
                        "Progress update",
                        session_id=session_id,
                        phase=phase,
                        percentage=percentage,
                        current=current,
                        total=total
                    )
                
                # Log detailed progress for debugging
                logger.debug(
                    "Progress callback executed",
                    session_id=session_id,
                    phase=phase,
                    current=current,
                    total=total,
                    percentage=percentage,
                    progress_type=progress_data.get("type") if isinstance(progress_data, dict) else "parameter_format"
                )
                
            except Exception as e:
                logger.warning(
                    "Progress callback failed",
                    session_id=session_id,
                    error=str(e),
                    progress_data_type=type(progress_data).__name__,
                    exc_info=True
                )
        
        return progress_callback