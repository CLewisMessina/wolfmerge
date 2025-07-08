# app/services/parallel_processing/performance_monitor.py
import time
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import statistics
import structlog

from .job_queue import DocumentJob
from .batch_processor import ProcessingResult

logger = structlog.get_logger()

@dataclass
class ProcessingMetrics:
    """Comprehensive processing performance metrics"""
    # Basic metrics
    total_documents: int = 0
    total_processing_time: float = 0.0
    avg_processing_time: float = 0.0
    min_processing_time: float = float('inf')
    max_processing_time: float = 0.0
    
    # Success metrics
    successful_documents: int = 0
    failed_documents: int = 0
    success_rate: float = 0.0
    
    # German compliance metrics
    german_documents: int = 0
    german_processing_time: float = 0.0
    german_avg_time: float = 0.0
    
    # Performance categories
    processing_times: List[float] = field(default_factory=list)
    complexity_scores: List[float] = field(default_factory=list)
    
    # Throughput metrics
    documents_per_minute: float = 0.0
    parallel_efficiency: float = 0.0
    
    # Quality metrics
    avg_chunks_per_document: float = 0.0
    avg_compliance_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "basic_metrics": {
                "total_documents": self.total_documents,
                "total_processing_time": self.total_processing_time,
                "avg_processing_time": self.avg_processing_time,
                "min_processing_time": self.min_processing_time if self.min_processing_time != float('inf') else 0.0,
                "max_processing_time": self.max_processing_time
            },
            "success_metrics": {
                "successful_documents": self.successful_documents,
                "failed_documents": self.failed_documents,
                "success_rate": self.success_rate
            },
            "german_metrics": {
                "german_documents": self.german_documents,
                "german_processing_time": self.german_processing_time,
                "german_avg_time": self.german_avg_time
            },
            "throughput_metrics": {
                "documents_per_minute": self.documents_per_minute,
                "parallel_efficiency": self.parallel_efficiency
            },
            "quality_metrics": {
                "avg_chunks_per_document": self.avg_chunks_per_document,
                "avg_compliance_score": self.avg_compliance_score
            }
        }

class PerformanceMonitor:
    """Monitor and analyze processing performance for optimization"""
    
    def __init__(self, history_limit: int = 1000):
        self.history_limit = history_limit
        
        # Performance tracking
        self.processing_history: List[Dict[str, Any]] = []
        self.session_metrics = ProcessingMetrics()
        
        # Benchmark targets (Day 3 goals)
        self.performance_targets = {
            'max_avg_processing_time': 3.0,  # 3 seconds per document
            'min_success_rate': 0.95,        # 95% success rate
            'min_throughput': 20.0,          # 20 documents per minute
            'max_batch_time': 60.0           # 60 seconds for 20 documents
        }
        
        # Real-time tracking
        self.current_batch_start = None
        self.current_batch_size = 0
    
    def start_batch_monitoring(self, batch_size: int):
        """Start monitoring a new batch"""
        self.current_batch_start = time.time()
        self.current_batch_size = batch_size
        
        logger.info(
            "Performance monitoring started",
            batch_size=batch_size,
            targets=self.performance_targets
        )
    
    def record_processing_results(self, results: List[ProcessingResult]):
        """Record processing results for performance analysis"""
        
        if not results:
            return
        
        batch_start_time = self.current_batch_start or time.time()
        batch_total_time = time.time() - batch_start_time
        
        # Process individual results
        for result in results:
            self._record_individual_result(result)
        
        # Calculate batch metrics
        batch_metrics = self._calculate_batch_metrics(results, batch_total_time)
        
        # Update session metrics
        self._update_session_metrics(results)
        
        # Add to history
        self._add_to_history(batch_metrics)
        
        # Log performance summary
        self._log_performance_summary(batch_metrics)
        
        # Check if targets are met
        self._check_performance_targets(batch_metrics)
    
    def _record_individual_result(self, result: ProcessingResult):
        """Record individual processing result"""
        
        if result.success and result.processing_time > 0:
            # Add to processing times for analysis
            self.session_metrics.processing_times.append(result.processing_time)
            self.session_metrics.complexity_scores.append(result.job.complexity_score)
        
        # Track German document performance
        if result.job.is_german_compliance:
            self.session_metrics.german_documents += 1
            if result.success:
                self.session_metrics.german_processing_time += result.processing_time
    
    def _calculate_batch_metrics(self, results: List[ProcessingResult], batch_time: float) -> Dict[str, Any]:
        """Calculate comprehensive batch performance metrics"""
        
        successful_results = [r for r in results if r.success]
        failed_results = [r for r in results if not r.success]
        
        # Basic calculations
        total_docs = len(results)
        processing_times = [r.processing_time for r in successful_results if r.processing_time > 0]
        
        # Calculate metrics
        metrics = {
            "batch_info": {
                "total_documents": total_docs,
                "batch_processing_time": batch_time,
                "successful_documents": len(successful_results),
                "failed_documents": len(failed_results)
            },
            "timing_metrics": {},
            "quality_metrics": {},
            "efficiency_metrics": {},
            "german_metrics": {}
        }
        
        # Timing metrics
        if processing_times:
            metrics["timing_metrics"] = {
                "avg_processing_time": statistics.mean(processing_times),
                "median_processing_time": statistics.median(processing_times),
                "min_processing_time": min(processing_times),
                "max_processing_time": max(processing_times),
                "std_dev_processing_time": statistics.stdev(processing_times) if len(processing_times) > 1 else 0.0
            }
        
        # Quality metrics
        chunks_created = [r.chunks_created for r in successful_results if r.chunks_created > 0]
        performance_scores = [r.performance_score for r in successful_results if r.performance_score > 0]
        
        if chunks_created:
            metrics["quality_metrics"]["avg_chunks_per_doc"] = statistics.mean(chunks_created)
        
        if performance_scores:
            metrics["quality_metrics"]["avg_performance_score"] = statistics.mean(performance_scores)
        
        # Efficiency metrics
        if batch_time > 0:
            metrics["efficiency_metrics"] = {
                "documents_per_minute": (total_docs / batch_time) * 60,
                "successful_docs_per_minute": (len(successful_results) / batch_time) * 60,
                "parallel_efficiency": self._calculate_parallel_efficiency(processing_times, batch_time),
                "success_rate": len(successful_results) / total_docs if total_docs > 0 else 0.0
            }
        
        # German compliance metrics
        german_results = [r for r in successful_results if r.job.is_german_compliance]
        if german_results:
            german_times = [r.processing_time for r in german_results if r.processing_time > 0]
            metrics["german_metrics"] = {
                "german_documents": len(german_results),
                "german_percentage": len(german_results) / total_docs * 100,
                "avg_german_processing_time": statistics.mean(german_times) if german_times else 0.0,
                "german_performance_bonus": statistics.mean([
                    r.performance_score for r in german_results if r.performance_score > 0
                ]) if german_results else 0.0
            }
        
        return metrics
    
    def _calculate_parallel_efficiency(self, processing_times: List[float], batch_time: float) -> float:
        """Calculate parallel processing efficiency"""
        
        if not processing_times or batch_time <= 0:
            return 0.0
        
        # Theoretical sequential time (sum of all processing times)
        sequential_time = sum(processing_times)
        
        # Parallel efficiency = sequential_time / (batch_time * num_documents)
        efficiency = sequential_time / (batch_time * len(processing_times))
        
        # Cap efficiency at 1.0 (100%) and convert to percentage
        return min(1.0, efficiency)
    
    def _update_session_metrics(self, results: List[ProcessingResult]):
        """Update session-level metrics"""
        
        for result in results:
            self.session_metrics.total_documents += 1
            self.session_metrics.total_processing_time += result.processing_time
            
            if result.success:
                self.session_metrics.successful_documents += 1
                
                # Update min/max times
                if result.processing_time > 0:
                    if result.processing_time < self.session_metrics.min_processing_time:
                        self.session_metrics.min_processing_time = result.processing_time
                    if result.processing_time > self.session_metrics.max_processing_time:
                        self.session_metrics.max_processing_time = result.processing_time
            else:
                self.session_metrics.failed_documents += 1
        
        # Calculate derived metrics
        if self.session_metrics.total_documents > 0:
            self.session_metrics.avg_processing_time = (
                self.session_metrics.total_processing_time / self.session_metrics.total_documents
            )
            self.session_metrics.success_rate = (
                self.session_metrics.successful_documents / self.session_metrics.total_documents
            )
        
        # German-specific metrics
        if self.session_metrics.german_documents > 0:
            self.session_metrics.german_avg_time = (
                self.session_metrics.german_processing_time / self.session_metrics.german_documents
            )
    
    def _add_to_history(self, batch_metrics: Dict[str, Any]):
        """Add batch metrics to processing history"""
        
        history_entry = {
            "timestamp": time.time(),
            "batch_metrics": batch_metrics,
            "session_snapshot": self.session_metrics.to_dict()
        }
        
        self.processing_history.append(history_entry)
        
        # Maintain history size limit
        if len(self.processing_history) > self.history_limit:
            self.processing_history = self.processing_history[-self.history_limit:]
    
    def _log_performance_summary(self, batch_metrics: Dict[str, Any]):
        """Log comprehensive performance summary"""
        
        timing = batch_metrics.get("timing_metrics", {})
        efficiency = batch_metrics.get("efficiency_metrics", {})
        
        logger.info(
            "Batch performance summary",
            total_docs=batch_metrics["batch_info"]["total_documents"],
            avg_time=timing.get("avg_processing_time", 0),
            success_rate=efficiency.get("success_rate", 0),
            docs_per_minute=efficiency.get("documents_per_minute", 0),
            parallel_efficiency=efficiency.get("parallel_efficiency", 0),
            german_docs=batch_metrics.get("german_metrics", {}).get("german_documents", 0)
        )
    
    def _check_performance_targets(self, batch_metrics: Dict[str, Any]):
        """Check if performance targets are being met"""
        
        timing = batch_metrics.get("timing_metrics", {})
        efficiency = batch_metrics.get("efficiency_metrics", {})
        
        avg_time = timing.get("avg_processing_time", float('inf'))
        success_rate = efficiency.get("success_rate", 0.0)
        throughput = efficiency.get("documents_per_minute", 0.0)
        batch_time = batch_metrics["batch_info"]["batch_processing_time"]
        
        # Check targets
        targets_met = {
            "avg_processing_time": avg_time <= self.performance_targets["max_avg_processing_time"],
            "success_rate": success_rate >= self.performance_targets["min_success_rate"],
            "throughput": throughput >= self.performance_targets["min_throughput"],
            "batch_time": batch_time <= self.performance_targets["max_batch_time"]
        }
        
        # Log performance grade
        grade = self._calculate_performance_grade(targets_met, avg_time, success_rate)
        
        if grade in ["A", "B"]:
            logger.info(f"Performance target achieved: Grade {grade}", targets_met=targets_met)
        else:
            logger.warning(f"Performance below target: Grade {grade}", targets_met=targets_met)
    
    def _calculate_performance_grade(self, targets_met: Dict[str, bool], avg_time: float, success_rate: float) -> str:
        """Calculate overall performance grade"""
        
        targets_achieved = sum(targets_met.values())
        
        # Grade based on targets met and specific metrics
        if targets_achieved == 4:
            return "A"
        elif targets_achieved >= 3 and success_rate >= 0.90:
            return "B"
        elif targets_achieved >= 2 and success_rate >= 0.80:
            return "C"
        elif targets_achieved >= 1 and success_rate >= 0.70:
            return "D"
        else:
            return "F"
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get current processing statistics"""
        
        return self.session_metrics.to_dict()
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        
        current_metrics = self.session_metrics.to_dict()
        
        # Calculate recent performance trends
        recent_history = self.processing_history[-10:] if len(self.processing_history) >= 10 else self.processing_history
        
        trends = {}
        if recent_history:
            recent_times = []
            recent_success_rates = []
            
            for entry in recent_history:
                timing = entry["batch_metrics"].get("timing_metrics", {})
                efficiency = entry["batch_metrics"].get("efficiency_metrics", {})
                
                if timing.get("avg_processing_time"):
                    recent_times.append(timing["avg_processing_time"])
                if efficiency.get("success_rate") is not None:
                    recent_success_rates.append(efficiency["success_rate"])
            
            if recent_times:
                trends["avg_time_trend"] = "improving" if len(recent_times) > 1 and recent_times[-1] < recent_times[0] else "stable"
            if recent_success_rates:
                trends["success_trend"] = "improving" if len(recent_success_rates) > 1 and recent_success_rates[-1] > recent_success_rates[0] else "stable"
        
        return {
            "current_metrics": current_metrics,
            "performance_targets": self.performance_targets,
            "targets_status": self._get_targets_status(),
            "trends": trends,
            "recommendations": self._generate_performance_recommendations(),
            "history_entries": len(self.processing_history)
        }
    
    def _get_targets_status(self) -> Dict[str, str]:
        """Get current status against performance targets"""
        
        metrics = self.session_metrics
        
        return {
            "avg_processing_time": "met" if metrics.avg_processing_time <= self.performance_targets["max_avg_processing_time"] else "not_met",
            "success_rate": "met" if metrics.success_rate >= self.performance_targets["min_success_rate"] else "not_met",
            "throughput": "met" if metrics.documents_per_minute >= self.performance_targets["min_throughput"] else "not_met"
        }
    
    def _generate_performance_recommendations(self) -> List[str]:
        """Generate performance optimization recommendations"""
        
        recommendations = []
        metrics = self.session_metrics
        
        # Processing time recommendations
        if metrics.avg_processing_time > self.performance_targets["max_avg_processing_time"]:
            recommendations.append(
                f"Average processing time ({metrics.avg_processing_time:.2f}s) exceeds target "
                f"({self.performance_targets['max_avg_processing_time']}s). Consider optimizing OpenAI prompts or increasing parallelism."
            )
        
        # Success rate recommendations
        if metrics.success_rate < self.performance_targets["min_success_rate"]:
            recommendations.append(
                f"Success rate ({metrics.success_rate:.1%}) below target "
                f"({self.performance_targets['min_success_rate']:.1%}). Review error patterns and improve error handling."
            )
        
        # German document performance
        if metrics.german_documents > 0 and metrics.german_avg_time > metrics.avg_processing_time * 1.2:
            recommendations.append(
                "German documents taking 20% longer than average. Consider optimizing German-specific processing."
            )
        
        # Throughput recommendations
        if metrics.documents_per_minute < self.performance_targets["min_throughput"]:
            recommendations.append(
                f"Throughput ({metrics.documents_per_minute:.1f} docs/min) below target "
                f"({self.performance_targets['min_throughput']} docs/min). Consider increasing parallel processing workers."
            )
        
        # Quality recommendations
        if metrics.avg_chunks_per_document < 3:
            recommendations.append(
                "Low average chunks per document. Verify Docling processing is working optimally."
            )
        
        if not recommendations:
            recommendations.append("Performance targets are being met. System operating optimally.")
        
        return recommendations
    
    def reset_session_metrics(self):
        """Reset session metrics for new analysis session"""
        
        self.session_metrics = ProcessingMetrics()
        self.current_batch_start = None
        self.current_batch_size = 0
        
        logger.info("Performance monitoring session reset")