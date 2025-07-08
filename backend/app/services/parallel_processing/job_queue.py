# app/services/parallel_processing/job_queue.py
import hashlib
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple, Dict, Optional
import structlog
from pathlib import Path

logger = structlog.get_logger()

class JobPriority(Enum):
    """Job priority levels for intelligent processing"""
    CRITICAL = 1    # German compliance docs, audit-critical
    HIGH = 2        # German docs, compliance-relevant
    MEDIUM = 3      # Non-German compliance docs
    LOW = 4         # Other documents

@dataclass
class DocumentJob:
    """Individual document processing job with intelligence metadata"""
    filename: str
    content: bytes
    size: int
    
    # Priority and processing metadata
    priority: JobPriority
    complexity_score: float
    processing_estimate_seconds: float
    
    # German compliance intelligence
    german_score: float
    language_detected: str
    compliance_indicators: List[str] = field(default_factory=list)
    
    # Processing tracking
    job_id: str = field(default_factory=lambda: hashlib.md5(str(time.time()).encode()).hexdigest()[:8])
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    
    def mark_started(self):
        """Mark job as started"""
        self.started_at = time.time()
    
    def mark_completed(self):
        """Mark job as completed"""
        self.completed_at = time.time()
    
    @property
    def processing_time(self) -> Optional[float]:
        """Get actual processing time if completed"""
        if self.started_at and self.completed_at:
            return self.completed_at - self.started_at
        return None
    
    @property
    def is_german_compliance(self) -> bool:
        """Check if this is a German compliance document"""
        return self.german_score > 0.3 and len(self.compliance_indicators) > 0

class JobQueue:
    """Intelligent job queue with German compliance prioritization"""
    
    def __init__(self, max_batch_size: int = 3, max_batch_complexity: float = 2.0):
        self.max_batch_size = max_batch_size
        self.max_batch_complexity = max_batch_complexity
        
        # German legal indicators for prioritization
        self.german_legal_terms = {
            'high_priority': [
                'dsgvo', 'datenschutzgrundverordnung', 'aufsichtsbehörde',
                'dsfa', 'datenschutz-folgenabschätzung', 'bfdi', 'baylda'
            ],
            'medium_priority': [
                'datenschutz', 'personenbezogene daten', 'verarbeitung',
                'einwilligung', 'betroffenenrechte', 'rechtsgrundlage'
            ],
            'compliance_indicators': [
                'verfahrensverzeichnis', 'privacy policy', 'datenschutzerklärung',
                'consent', 'einwilligung', 'artikel', 'art.', 'gdpr'
            ]
        }
        
        # File type complexity mappings
        self.file_complexity = {
            '.pdf': 1.0,
            '.docx': 0.8,
            '.doc': 0.8,
            '.txt': 0.3,
            '.md': 0.2
        }
    
    def create_job_queue(self, files: List[Tuple[str, bytes, int]], workspace_id: str) -> List[DocumentJob]:
        """Create intelligent job queue with German compliance prioritization"""
        
        jobs = []
        
        logger.info(
            "Creating job queue",
            workspace_id=workspace_id,
            file_count=len(files)
        )
        
        for filename, content, size in files:
            job = self._create_document_job(filename, content, size)
            jobs.append(job)
        
        # Sort by priority and complexity for optimal processing
        jobs.sort(key=lambda x: (
            x.priority.value,           # German compliance first
            -x.complexity_score,        # Complex docs first within priority
            x.size                      # Smaller docs first within complexity
        ))
        
        logger.info(
            "Job queue created",
            total_jobs=len(jobs),
            german_docs=sum(1 for job in jobs if job.is_german_compliance),
            priority_breakdown={
                priority.name: sum(1 for job in jobs if job.priority == priority)
                for priority in JobPriority
            }
        )
        
        return jobs
    
    def _create_document_job(self, filename: str, content: bytes, size: int) -> DocumentJob:
        """Create individual document job with intelligence analysis"""
        
        # Quick content preview for analysis
        try:
            text_preview = content.decode('utf-8', errors='ignore')[:2000]
        except Exception:
            text_preview = ""
        
        # Detect German compliance content
        german_score = self._calculate_german_score(text_preview, filename)
        language = self._detect_language(text_preview, filename, german_score)
        compliance_indicators = self._detect_compliance_indicators(text_preview, filename)
        
        # Calculate processing complexity
        complexity = self._calculate_complexity(content, filename, text_preview)
        
        # Determine priority
        priority = self._determine_priority(german_score, compliance_indicators, complexity)
        
        # Estimate processing time (for UI progress)
        processing_estimate = self._estimate_processing_time(complexity, priority, size)
        
        job = DocumentJob(
            filename=filename,
            content=content,
            size=size,
            priority=priority,
            complexity_score=complexity,
            processing_estimate_seconds=processing_estimate,
            german_score=german_score,
            language_detected=language,
            compliance_indicators=compliance_indicators
        )
        
        logger.debug(
            "Document job created",
            filename=filename,
            priority=priority.name,
            german_score=german_score,
            complexity=complexity,
            estimate_seconds=processing_estimate
        )
        
        return job
    
    def _calculate_german_score(self, text: str, filename: str) -> float:
        """Calculate German content score for prioritization"""
        
        text_lower = text.lower()
        filename_lower = filename.lower()
        
        score = 0.0
        total_checks = 0
        
        # Check high priority German terms
        for term in self.german_legal_terms['high_priority']:
            total_checks += 1
            if term in text_lower:
                score += 3.0
            if term in filename_lower:
                score += 4.0  # Filename carries more weight
        
        # Check medium priority terms
        for term in self.german_legal_terms['medium_priority']:
            total_checks += 1
            if term in text_lower:
                score += 1.5
            if term in filename_lower:
                score += 2.0
        
        # Check compliance indicators
        for term in self.german_legal_terms['compliance_indicators']:
            total_checks += 1
            if term in text_lower:
                score += 1.0
            if term in filename_lower:
                score += 1.5
        
        # Normalize score
        if total_checks > 0:
            normalized_score = min(1.0, score / (total_checks * 2))
        else:
            normalized_score = 0.0
        
        return normalized_score
    
    def _detect_language(self, text: str, filename: str, german_score: float) -> str:
        """Detect document language"""
        
        if german_score > 0.4:
            return "de"
        elif german_score > 0.1:
            return "mixed"
        else:
            # Simple English detection
            english_indicators = ['the', 'and', 'for', 'with', 'data', 'privacy', 'policy']
            english_count = sum(1 for word in english_indicators if word in text.lower())
            
            if english_count > 3:
                return "en"
            else:
                return "unknown"
    
    def _detect_compliance_indicators(self, text: str, filename: str) -> List[str]:
        """Detect compliance-relevant indicators in document"""
        
        indicators = []
        text_lower = text.lower()
        filename_lower = filename.lower()
        
        # Document type indicators
        doc_type_patterns = {
            'privacy_policy': ['privacy policy', 'datenschutzerklärung', 'data protection notice'],
            'dsfa': ['dsfa', 'dpia', 'impact assessment', 'folgenabschätzung'],
            'ropa': ['ropa', 'records of processing', 'verfahrensverzeichnis'],
            'consent_form': ['consent', 'einwilligung', 'opt-in'],
            'contract': ['contract', 'vertrag', 'agreement', 'vereinbarung'],
            'policy': ['policy', 'richtlinie', 'guideline'],
            'training': ['training', 'schulung', 'awareness']
        }
        
        for doc_type, patterns in doc_type_patterns.items():
            if any(pattern in text_lower or pattern in filename_lower for pattern in patterns):
                indicators.append(doc_type)
        
        # GDPR article references
        gdpr_patterns = [
            'art. 5', 'art. 6', 'art. 7', 'art. 13', 'art. 14', 'art. 15',
            'art. 16', 'art. 17', 'art. 18', 'art. 20', 'art. 25', 'art. 30',
            'art. 32', 'art. 35', 'artikel 5', 'artikel 6'
        ]
        
        for pattern in gdpr_patterns:
            if pattern in text_lower:
                indicators.append(f'gdpr_{pattern.replace(".", "_").replace(" ", "_")}')
        
        # German authority references
        authority_patterns = ['bfdi', 'baylda', 'lfd', 'aufsichtsbehörde']
        for pattern in authority_patterns:
            if pattern in text_lower:
                indicators.append(f'authority_{pattern}')
        
        return indicators
    
    def _calculate_complexity(self, content: bytes, filename: str, text_preview: str) -> float:
        """Calculate document processing complexity"""
        
        complexity = 0.0
        
        # Size-based complexity
        size_mb = len(content) / (1024 * 1024)
        complexity += size_mb * 0.3
        
        # File type complexity
        file_ext = Path(filename).suffix.lower()
        complexity += self.file_complexity.get(file_ext, 0.5)
        
        # Content complexity indicators
        if len(text_preview) > 0:
            # Word count complexity
            word_count = len(text_preview.split())
            complexity += min(1.0, word_count / 10000)  # Max 1.0 for word count
            
            # Structure complexity (tables, lists)
            structure_indicators = text_preview.count('\n') + text_preview.count('\t')
            complexity += min(0.5, structure_indicators / 100)
            
            # Technical complexity (legal references, technical terms)
            technical_terms = ['art.', 'artikel', 'section', 'clause', 'annex']
            technical_count = sum(text_preview.lower().count(term) for term in technical_terms)
            complexity += min(0.3, technical_count / 20)
        
        return min(3.0, complexity)  # Cap at 3.0 max complexity
    
    def _determine_priority(self, german_score: float, compliance_indicators: List[str], complexity: float) -> JobPriority:
        """Determine job priority based on content analysis"""
        
        # Critical priority for high-value German compliance docs
        if german_score > 0.6 and any('dsfa' in indicator or 'ropa' in indicator for indicator in compliance_indicators):
            return JobPriority.CRITICAL
        
        # High priority for German compliance content
        if german_score > 0.3 and len(compliance_indicators) > 2:
            return JobPriority.HIGH
        
        # Medium priority for compliance-relevant content
        if len(compliance_indicators) > 0 or german_score > 0.1:
            return JobPriority.MEDIUM
        
        # Low priority for everything else
        return JobPriority.LOW
    
    def _estimate_processing_time(self, complexity: float, priority: JobPriority, size: int) -> float:
        """Estimate processing time for progress tracking"""
        
        # Base time estimates (seconds)
        base_times = {
            JobPriority.CRITICAL: 8.0,  # More thorough analysis
            JobPriority.HIGH: 6.0,
            JobPriority.MEDIUM: 4.0,
            JobPriority.LOW: 3.0
        }
        
        base_time = base_times[priority]
        
        # Adjust for complexity
        complexity_multiplier = 0.5 + (complexity / 3.0)  # 0.5x to 1.5x based on complexity
        
        # Adjust for size (very rough estimate)
        size_multiplier = 1.0 + min(1.0, size / (1024 * 1024))  # +1x per MB
        
        estimated_time = base_time * complexity_multiplier * size_multiplier
        
        return min(30.0, estimated_time)  # Cap at 30 seconds per document
    
    def create_intelligent_batches(self, jobs: List[DocumentJob]) -> List[List[DocumentJob]]:
        """Create optimized batches for parallel processing"""
        
        batches = []
        current_batch = []
        current_complexity = 0.0
        
        for job in jobs:
            # Check if adding this job would exceed batch limits
            would_exceed_size = len(current_batch) >= self.max_batch_size
            would_exceed_complexity = current_complexity + job.complexity_score > self.max_batch_complexity
            
            # Start new batch if limits would be exceeded
            if current_batch and (would_exceed_size or would_exceed_complexity):
                batches.append(current_batch)
                current_batch = []
                current_complexity = 0.0
            
            # Add job to current batch
            current_batch.append(job)
            current_complexity += job.complexity_score
        
        # Add final batch if not empty
        if current_batch:
            batches.append(current_batch)
        
        logger.info(
            "Intelligent batches created",
            total_batches=len(batches),
            avg_batch_size=sum(len(batch) for batch in batches) / len(batches) if batches else 0,
            avg_complexity=sum(sum(job.complexity_score for job in batch) for batch in batches) / len(batches) if batches else 0
        )
        
        return batches