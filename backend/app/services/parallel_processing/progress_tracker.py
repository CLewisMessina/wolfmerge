# app/services/parallel_processing/progress_tracker.py
import asyncio
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any
from enum import Enum
import structlog

logger = structlog.get_logger()

class ProgressStatus(Enum):
    """Progress status types"""
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    ERROR = "error"
    CANCELLED = "cancelled"

@dataclass
class ProgressUpdate:
    """Individual progress update for a document job"""
    job_id: str
    filename: str
    status: ProgressStatus
    
    # Progress metrics
    progress_percentage: float = 0.0
    processing_time: float = 0.0
    estimated_remaining: float = 0.0
    
    # Job metadata
    batch_info: Optional[str] = None
    priority: str = "medium"
    german_detected: bool = False
    
    # Analysis results
    chunks_created: int = 0
    performance_score: float = 0.0
    compliance_indicators: List[str] = field(default_factory=list)
    
    # Error information
    error_message: Optional[str] = None
    
    # Timestamps
    started_at: Optional[float] = None
    updated_at: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "job_id": self.job_id,
            "filename": self.filename,
            "status": self.status.value,
            "progress_percentage": self.progress_percentage,
            "processing_time": self.processing_time,
            "estimated_remaining": self.estimated_remaining,
            "batch_info": self.batch_info,
            "priority": self.priority,
            "german_detected": self.german_detected,
            "chunks_created": self.chunks_created,
            "performance_score": self.performance_score,
            "compliance_indicators": self.compliance_indicators,
            "error_message": self.error_message,
            "started_at": self.started_at,
            "updated_at": self.updated_at
        }

@dataclass
class BatchProgress:
    """Overall batch processing progress"""
    workspace_id: str
    total_jobs: int
    completed_jobs: int = 0
    failed_jobs: int = 0
    
    # Timing
    started_at: float = field(default_factory=time.time)
    estimated_completion: Optional[float] = None
    
    # Performance metrics
    total_processing_time: float = 0.0
    avg_processing_time: float = 0.0
    german_docs_processed: int = 0
    total_chunks_created: int = 0
    
    @property
    def progress_percentage(self) -> float:
        """Calculate overall progress percentage"""
        if self.total_jobs == 0:
            return 0.0
        return (self.completed_jobs + self.failed_jobs) / self.total_jobs * 100
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate"""
        total_processed = self.completed_jobs + self.failed_jobs
        if total_processed == 0:
            return 0.0
        return self.completed_jobs / total_processed * 100
    
    @property
    def elapsed_time(self) -> float:
        """Get elapsed processing time"""
        return time.time() - self.started_at
    
    def estimate_completion_time(self) -> Optional[float]:
        """Estimate completion time based on current progress"""
        if self.completed_jobs == 0 or self.avg_processing_time == 0:
            return None
        
        remaining_jobs = self.total_jobs - self.completed_jobs - self.failed_jobs
        estimated_remaining_time = remaining_jobs * self.avg_processing_time
        
        return time.time() + estimated_remaining_time
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "workspace_id": self.workspace_id,
            "total_jobs": self.total_jobs,
            "completed_jobs": self.completed_jobs,
            "failed_jobs": self.failed_jobs,
            "progress_percentage": self.progress_percentage,
            "success_rate": self.success_rate,
            "elapsed_time": self.elapsed_time,
            "estimated_completion": self.estimate_completion_time(),
            "total_processing_time": self.total_processing_time,
            "avg_processing_time": self.avg_processing_time,
            "german_docs_processed": self.german_docs_processed,
            "total_chunks_created": self.total_chunks_created,
            "started_at": self.started_at
        }

class ProgressTracker:
    """Real-time progress tracking for batch processing"""
    
    def __init__(self):
        # Active progress tracking
        self.job_progress: Dict[str, ProgressUpdate] = {}
        self.batch_progress: Dict[str, BatchProgress] = {}
        
        # WebSocket callback registry
        self.progress_callbacks: Dict[str, List[Callable]] = {}
        
        # Progress history for analytics
        self.progress_history: List[Dict[str, Any]] = []
        self.max_history_size = 1000
    
    def register_callback(self, workspace_id: str, callback: Callable):
        """Register a progress callback for workspace"""
        if workspace_id not in self.progress_callbacks:
            self.progress_callbacks[workspace_id] = []
        
        self.progress_callbacks[workspace_id].append(callback)
        
        logger.debug(
            "Progress callback registered",
            workspace_id=workspace_id,
            total_callbacks=len(self.progress_callbacks[workspace_id])
        )
    
    def unregister_callback(self, workspace_id: str, callback: Callable):
        """Unregister a progress callback"""
        if workspace_id in self.progress_callbacks:
            try:
                self.progress_callbacks[workspace_id].remove(callback)
                if not self.progress_callbacks[workspace_id]:
                    del self.progress_callbacks[workspace_id]
            except ValueError:
                pass  # Callback not found
    
    def start_batch_tracking(self, workspace_id: str, total_jobs: int):
        """Start tracking a new batch"""
        batch_progress = BatchProgress(
            workspace_id=workspace_id,
            total_jobs=total_jobs
        )
        
        self.batch_progress[workspace_id] = batch_progress
        
        logger.info(
            "Batch tracking started",
            workspace_id=workspace_id,
            total_jobs=total_jobs
        )
        
        # Notify callbacks
        asyncio.create_task(self._notify_callbacks(workspace_id, {
            "type": "batch_started",
            "batch_progress": batch_progress.to_dict()
        }))
    
    async def update_job_progress(self, workspace_id: str, progress_data: Dict[str, Any]):
        """Update progress for individual job"""
        
        job_id = progress_data.get("job_id")
        if not job_id:
            logger.warning("Progress update missing job_id", workspace_id=workspace_id)
            return
        
        # Parse status
        status_str = progress_data.get("status", "processing")
        try:
            status = ProgressStatus(status_str)
        except ValueError:
            status = ProgressStatus.PROCESSING
        
        # Create or update progress
        if job_id in self.job_progress:
            progress = self.job_progress[job_id]
            progress.status = status
            progress.updated_at = time.time()
        else:
            progress = ProgressUpdate(
                job_id=job_id,
                filename=progress_data.get("filename", "unknown"),
                status=status,
                started_at=time.time() if status == ProgressStatus.PROCESSING else None
            )
            self.job_progress[job_id] = progress
        
        # Update progress fields
        self._update_progress_fields(progress, progress_data)
        
        # Update batch progress
        await self._update_batch_progress(workspace_id, progress)
        
        # Notify callbacks
        await self._notify_callbacks(workspace_id, {
            "type": "job_progress",
            "job_progress": progress.to_dict(),
            "batch_progress": self.batch_progress.get(workspace_id, {}).to_dict() if workspace_id in self.batch_progress else {}
        })
        
        # Add to history
        self._add_to_history(workspace_id, progress)
    
    def _update_progress_fields(self, progress: ProgressUpdate, data: Dict[str, Any]):
        """Update progress fields from data"""
        
        # Basic fields
        if "processing_time" in data:
            progress.processing_time = float(data["processing_time"])
        
        if "batch" in data:
            progress.batch_info = data["batch"]
        
        if "priority" in data:
            progress.priority = data["priority"]
        
        if "german_detected" in data:
            progress.german_detected = bool(data["german_detected"])
        
        if "chunks_created" in data:
            progress.chunks_created = int(data["chunks_created"])
        
        if "performance_score" in data:
            progress.performance_score = float(data["performance_score"])
        
        if "error" in data:
            progress.error_message = data["error"]
        
        # Calculate progress percentage based on status
        if progress.status == ProgressStatus.QUEUED:
            progress.progress_percentage = 0.0
        elif progress.status == ProgressStatus.PROCESSING:
            progress.progress_percentage = 50.0
        elif progress.status in [ProgressStatus.COMPLETED, ProgressStatus.ERROR]:
            progress.progress_percentage = 100.0
        
        # Estimate remaining time
        if progress.status == ProgressStatus.PROCESSING and progress.started_at:
            elapsed = time.time() - progress.started_at
            if elapsed > 0:
                # Simple estimation: if we're 50% done, we need the same time to complete
                progress.estimated_remaining = elapsed
    
    async def _update_batch_progress(self, workspace_id: str, job_progress: ProgressUpdate):
        """Update batch-level progress based on job completion"""
        
        if workspace_id not in self.batch_progress:
            return
        
        batch = self.batch_progress[workspace_id]
        
        # Count completed and failed jobs
        completed_count = sum(
            1 for p in self.job_progress.values()
            if p.status == ProgressStatus.COMPLETED
        )
        
        failed_count = sum(
            1 for p in self.job_progress.values()
            if p.status == ProgressStatus.ERROR
        )
        
        # Update batch metrics
        batch.completed_jobs = completed_count
        batch.failed_jobs = failed_count
        
        # Calculate performance metrics
        completed_jobs = [
            p for p in self.job_progress.values()
            if p.status == ProgressStatus.COMPLETED
        ]
        
        if completed_jobs:
            batch.total_processing_time = sum(p.processing_time for p in completed_jobs)
            batch.avg_processing_time = batch.total_processing_time / len(completed_jobs)
            batch.german_docs_processed = sum(1 for p in completed_jobs if p.german_detected)
            batch.total_chunks_created = sum(p.chunks_created for p in completed_jobs)
        
        # Update estimated completion
        batch.estimated_completion = batch.estimate_completion_time()
        
        logger.debug(
            "Batch progress updated",
            workspace_id=workspace_id,
            progress_percentage=batch.progress_percentage,
            completed_jobs=batch.completed_jobs,
            failed_jobs=batch.failed_jobs
        )
    
    async def _notify_callbacks(self, workspace_id: str, message: Dict[str, Any]):
        """Notify all registered callbacks for workspace"""
        
        callbacks = self.progress_callbacks.get(workspace_id, [])
        
        if not callbacks:
            return
        
        # Add timestamp to message
        message["timestamp"] = time.time()
        
        # Call all callbacks (remove failed ones)
        failed_callbacks = []
        for callback in callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(message)
                else:
                    callback(message)
            except Exception as e:
                logger.warning(
                    "Progress callback failed",
                    workspace_id=workspace_id,
                    error=str(e)
                )
                failed_callbacks.append(callback)
        
        # Remove failed callbacks
        for failed_callback in failed_callbacks:
            self.unregister_callback(workspace_id, failed_callback)
    
    def _add_to_history(self, workspace_id: str, progress: ProgressUpdate):
        """Add progress update to history"""
        
        history_entry = {
            "workspace_id": workspace_id,
            "timestamp": time.time(),
            "progress": progress.to_dict()
        }
        
        self.progress_history.append(history_entry)
        
        # Maintain history size limit
        if len(self.progress_history) > self.max_history_size:
            self.progress_history = self.progress_history[-self.max_history_size:]
    
    def get_workspace_progress(self, workspace_id: str) -> Optional[Dict[str, Any]]:
        """Get current progress for workspace"""
        
        if workspace_id not in self.batch_progress:
            return None
        
        batch = self.batch_progress[workspace_id]
        
        # Get job progress for this workspace
        workspace_jobs = [
            p.to_dict() for p in self.job_progress.values()
        ]
        
        return {
            "batch_progress": batch.to_dict(),
            "job_progress": workspace_jobs,
            "active_callbacks": len(self.progress_callbacks.get(workspace_id, []))
        }
    
    def cleanup_workspace(self, workspace_id: str):
        """Clean up tracking data for completed workspace"""
        
        # Remove batch progress
        if workspace_id in self.batch_progress:
            del self.batch_progress[workspace_id]
        
        # Remove callbacks
        if workspace_id in self.progress_callbacks:
            del self.progress_callbacks[workspace_id]
        
        # Clean up job progress (keep recent for analytics)
        # Only remove jobs older than 1 hour
        current_time = time.time()
        old_jobs = [
            job_id for job_id, progress in self.job_progress.items()
            if current_time - progress.updated_at > 3600  # 1 hour
        ]
        
        for job_id in old_jobs:
            del self.job_progress[job_id]
        
        logger.info(
            "Workspace tracking cleaned up",
            workspace_id=workspace_id,
            old_jobs_removed=len(old_jobs)
        )
    
    def get_performance_analytics(self) -> Dict[str, Any]:
        """Get performance analytics from tracking data"""
        
        current_time = time.time()
        recent_cutoff = current_time - 3600  # Last hour
        
        # Recent progress updates
        recent_progress = [
            entry for entry in self.progress_history
            if entry["timestamp"] > recent_cutoff
        ]
        
        # Calculate analytics
        analytics = {
            "active_workspaces": len(self.batch_progress),
            "active_jobs": len(self.job_progress),
            "recent_updates": len(recent_progress),
            "processing_stats": {}
        }
        
        if recent_progress:
            completed_jobs = [
                entry for entry in recent_progress
                if entry["progress"]["status"] == "completed"
            ]
            
            if completed_jobs:
                processing_times = [
                    job["progress"]["processing_time"]
                    for job in completed_jobs
                    if job["progress"]["processing_time"] > 0
                ]
                
                if processing_times:
                    analytics["processing_stats"] = {
                        "avg_processing_time": sum(processing_times) / len(processing_times),
                        "min_processing_time": min(processing_times),
                        "max_processing_time": max(processing_times),
                        "total_jobs_completed": len(completed_jobs),
                        "german_docs_percentage": sum(
                            1 for job in completed_jobs
                            if job["progress"]["german_detected"]
                        ) / len(completed_jobs) * 100
                    }
        
        return analytics

# Global progress tracker instance
progress_tracker = ProgressTracker()