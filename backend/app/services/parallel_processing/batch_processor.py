# app/services/parallel_processing/batch_processor.py
import asyncio
import time
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Callable
import structlog
from openai import OpenAI

from app.models.compliance import DocumentAnalysis, ComplianceFramework
from app.services.docling_processor import DoclingProcessor
from app.config import settings
from .job_queue import DocumentJob

logger = structlog.get_logger()

@dataclass
class ProcessingResult:
    """Result of document processing with performance metrics"""
    job: DocumentJob
    analysis: Optional[DocumentAnalysis]
    success: bool
    error_message: Optional[str] = None
    processing_time: float = 0.0
    chunks_created: int = 0
    german_terms_found: int = 0
    
    @property
    def performance_score(self) -> float:
        """Calculate processing performance score"""
        if not self.success:
            return 0.0
        
        # Base score for successful processing
        score = 0.5
        
        # Bonus for German compliance detection
        if self.job.is_german_compliance:
            score += 0.2
        
        # Bonus for fast processing
        if self.processing_time < self.job.processing_estimate_seconds:
            score += 0.2
        
        # Bonus for chunk creation (document intelligence)
        if self.chunks_created > 5:
            score += 0.1
        
        return min(1.0, score)

class BatchProcessor:
    """High-performance async batch processor with OpenAI rate limiting"""
    
    def __init__(self, max_concurrent_openai: int = 3, max_concurrent_docling: int = 5):
        self.max_concurrent_openai = max_concurrent_openai
        self.max_concurrent_docling = max_concurrent_docling
        
        # Rate limiting semaphores
        self.openai_semaphore = asyncio.Semaphore(max_concurrent_openai)
        self.docling_semaphore = asyncio.Semaphore(max_concurrent_docling)
        
        # Initialize processors
        self.openai_client = OpenAI(api_key=settings.openai_api_key)
        self.docling_processor = DoclingProcessor()
        
        # Performance tracking
        self.processing_stats = {
            'total_processed': 0,
            'total_time': 0.0,
            'errors': 0,
            'german_docs': 0,
            'chunks_created': 0
        }
    
    async def process_batches(
        self,
        batches: List[List[DocumentJob]],
        workspace_id: str,
        user_id: str,
        framework: ComplianceFramework,
        progress_callback: Optional[Callable] = None
    ) -> List[ProcessingResult]:
        """Process multiple batches in sequence with parallel jobs within each batch"""
        
        start_time = time.time()
        all_results = []
        total_batches = len(batches)
        
        logger.info(
            "Starting batch processing",
            workspace_id=workspace_id,
            total_batches=total_batches,
            total_jobs=sum(len(batch) for batch in batches)
        )
        
        for batch_idx, batch in enumerate(batches):
            logger.info(
                "Processing batch",
                batch_index=batch_idx + 1,
                total_batches=total_batches,
                batch_size=len(batch)
            )
            
            # Process batch in parallel
            batch_results = await self._process_single_batch(
                batch, workspace_id, user_id, framework, 
                progress_callback, batch_idx, total_batches
            )
            
            all_results.extend(batch_results)
            
            # Update global stats
            self._update_processing_stats(batch_results)
            
            # Brief pause between batches to avoid overwhelming services
            if batch_idx < total_batches - 1:
                await asyncio.sleep(0.5)
        
        total_time = time.time() - start_time
        
        logger.info(
            "Batch processing completed",
            workspace_id=workspace_id,
            total_results=len(all_results),
            total_time=total_time,
            avg_time_per_doc=total_time / len(all_results) if all_results else 0,
            success_rate=sum(1 for r in all_results if r.success) / len(all_results) if all_results else 0
        )
        
        return all_results
    
    async def _process_single_batch(
        self,
        batch: List[DocumentJob],
        workspace_id: str,
        user_id: str,
        framework: ComplianceFramework,
        progress_callback: Optional[Callable],
        batch_idx: int,
        total_batches: int
    ) -> List[ProcessingResult]:
        """Process single batch of jobs in parallel"""
        
        # Create processing tasks for all jobs in batch
        tasks = []
        for job in batch:
            task = self._process_single_job(
                job, workspace_id, user_id, framework,
                progress_callback, batch_idx, total_batches
            )
            tasks.append(task)
        
        # Execute all jobs in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert exceptions to error results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                job = batch[i]
                error_result = ProcessingResult(
                    job=job,
                    analysis=None,
                    success=False,
                    error_message=str(result),
                    processing_time=0.0
                )
                processed_results.append(error_result)
                
                logger.error(
                    "Job processing failed",
                    job_id=job.job_id,
                    filename=job.filename,
                    error=str(result)
                )
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def _process_single_job(
        self,
        job: DocumentJob,
        workspace_id: str,
        user_id: str,
        framework: ComplianceFramework,
        progress_callback: Optional[Callable],
        batch_idx: int,
        total_batches: int
    ) -> ProcessingResult:
        """Process single document job with rate limiting"""
        
        start_time = time.time()
        job.mark_started()
        
        try:
            # Update progress
            if progress_callback:
                await progress_callback({
                    "job_id": job.job_id,
                    "filename": job.filename,
                    "status": "processing",
                    "batch": f"{batch_idx + 1}/{total_batches}",
                    "priority": job.priority.name.lower(),
                    "german_detected": job.is_german_compliance
                })
            
            # Step 1: Process with Docling (can run in parallel)
            chunks, metadata = await self._process_with_docling(job)
            
            # Step 2: Analyze with OpenAI (rate limited)
            analysis = await self._analyze_with_openai(
                job, chunks, metadata, framework
            )
            
            job.mark_completed()
            processing_time = time.time() - start_time
            
            # Create successful result
            result = ProcessingResult(
                job=job,
                analysis=analysis,
                success=True,
                processing_time=processing_time,
                chunks_created=len(chunks),
                german_terms_found=len(metadata.get('german_terms_detected', []))
            )
            
            # Update progress
            if progress_callback:
                await progress_callback({
                    "job_id": job.job_id,
                    "filename": job.filename,
                    "status": "completed",
                    "processing_time": processing_time,
                    "performance_score": result.performance_score,
                    "chunks_created": len(chunks),
                    "german_content": metadata.get('german_content_detected', False)
                })
            
            logger.debug(
                "Job completed successfully",
                job_id=job.job_id,
                filename=job.filename,
                processing_time=processing_time,
                chunks_created=len(chunks)
            )
            
            return result
            
        except Exception as e:
            job.mark_completed()
            processing_time = time.time() - start_time
            
            # Create error result
            result = ProcessingResult(
                job=job,
                analysis=None,
                success=False,
                error_message=str(e),
                processing_time=processing_time
            )
            
            # Update progress
            if progress_callback:
                await progress_callback({
                    "job_id": job.job_id,
                    "filename": job.filename,
                    "status": "error",
                    "error": str(e),
                    "processing_time": processing_time
                })
            
            logger.error(
                "Job processing failed",
                job_id=job.job_id,
                filename=job.filename,
                error=str(e),
                processing_time=processing_time
            )
            
            return result
    
    async def _process_with_docling(self, job: DocumentJob) -> tuple[List[Dict], Dict[str, Any]]:
        """Process document with Docling intelligence"""
        
        async with self.docling_semaphore:
            try:
                chunks, metadata = await self.docling_processor.process_document(
                    job.content, job.filename, "workspace_id", "user_id"
                )
                
                logger.debug(
                    "Docling processing completed",
                    job_id=job.job_id,
                    chunks_created=len(chunks),
                    german_detected=metadata.get('german_content_detected', False)
                )
                
                return chunks, metadata
                
            except Exception as e:
                logger.warning(
                    "Docling processing failed, using fallback",
                    job_id=job.job_id,
                    error=str(e)
                )
                
                # Fallback to simple text processing
                try:
                    text_content = job.content.decode('utf-8', errors='ignore')
                except:
                    text_content = "Failed to decode document content"
                
                fallback_chunks = [{
                    'chunk_index': 0,
                    'content': text_content[:2000],  # First 2000 chars
                    'chunk_type': 'fallback_text',
                    'german_terms': {},
                    'compliance_tags': job.compliance_indicators
                }]
                
                fallback_metadata = {
                    'filename': job.filename,
                    'processing_method': 'fallback',
                    'german_content_detected': job.is_german_compliance,
                    'language_detected': job.language_detected
                }
                
                return fallback_chunks, fallback_metadata
    
    async def _analyze_with_openai(
        self,
        job: DocumentJob,
        chunks: List[Dict],
        metadata: Dict[str, Any],
        framework: ComplianceFramework
    ) -> DocumentAnalysis:
        """Analyze document with OpenAI using rate limiting"""
        
        async with self.openai_semaphore:
            try:
                # Create analysis prompt based on document intelligence
                prompt = self._create_analysis_prompt(job, chunks, framework)
                
                # Call OpenAI
                response = self.openai_client.chat.completions.create(
                    model=settings.openai_model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=500,
                    temperature=0.2
                )
                
                analysis_text = response.choices[0].message.content
                
                # Convert to DocumentAnalysis structure
                analysis = self._create_document_analysis(
                    job, chunks, metadata, analysis_text, framework
                )
                
                logger.debug(
                    "OpenAI analysis completed",
                    job_id=job.job_id,
                    response_length=len(analysis_text)
                )
                
                return analysis
                
            except Exception as e:
                logger.error(
                    "OpenAI analysis failed",
                    job_id=job.job_id,
                    error=str(e)
                )
                
                # Create fallback analysis
                return self._create_fallback_analysis(job, metadata, framework)
    
    def _create_analysis_prompt(
        self,
        job: DocumentJob,
        chunks: List[Dict],
        framework: ComplianceFramework
    ) -> str:
        """Create optimized analysis prompt based on document intelligence"""
        
        # Get first chunk for analysis (or combine if small)
        if chunks:
            content_preview = chunks[0].get('content', '')[:1500]
            german_terms = chunks[0].get('german_terms', {})
            compliance_tags = chunks[0].get('compliance_tags', [])
        else:
            content_preview = "No content available"
            german_terms = {}
            compliance_tags = []
        
        # Build framework-specific context
        framework_context = self._get_framework_context(framework, job.is_german_compliance)
        
        # Build German context if applicable
        german_context = ""
        if job.is_german_compliance:
            german_context = f"""
GERMAN DSGVO CONTEXT:
- Document Language: {job.language_detected}
- German Legal Terms Detected: {german_terms}
- Compliance Indicators: {job.compliance_indicators}
- Priority Level: {job.priority.name}

This is a German compliance document. Provide analysis with DSGVO expertise.
"""
        
        prompt = f"""You are an expert compliance analyst specializing in {framework.value.upper()}.

DOCUMENT: {job.filename}
TYPE: {', '.join(job.compliance_indicators) if job.compliance_indicators else 'General Document'}

{german_context}

CONTENT TO ANALYZE:
{content_preview}

{framework_context}

ANALYSIS REQUIREMENTS:
1. Identify specific compliance requirements and gaps
2. Detect risk indicators and violations
3. Provide actionable recommendations
4. Focus on German legal requirements if applicable

Provide concise, structured analysis for compliance professionals."""
        
        return prompt
    
    def _get_framework_context(self, framework: ComplianceFramework, is_german: bool) -> str:
        """Get framework-specific analysis context"""
        
        if framework == ComplianceFramework.GDPR:
            context = """
GDPR/DSGVO ANALYSIS FOCUS:
- Legal basis (Article 6) documentation
- Data subject rights (Articles 15-22) procedures
- Privacy by design (Article 25) implementation
- Security measures (Article 32) adequacy
- Records of processing (Article 30) completeness
- DPIA requirements (Article 35) when needed
"""
            if is_german:
                context += "\nApply German DSGVO terminology and BfDI/LfDI authority requirements."
        
        elif framework == ComplianceFramework.SOC2:
            context = """
SOC 2 TRUST SERVICE CRITERIA:
- Security: Protection against unauthorized access
- Availability: System operations and monitoring
- Processing Integrity: Completeness, validity, accuracy
- Confidentiality: Protection of confidential information
- Privacy: Collection, use, retention, disclosure practices
"""
        
        elif framework == ComplianceFramework.HIPAA:
            context = """
HIPAA SAFEGUARDS ANALYSIS:
- Administrative: Workforce training, access management
- Physical: Facility access, workstation controls
- Technical: Access controls, audit controls, encryption
"""
        
        else:
            context = "General compliance analysis focusing on control implementation."
        
        return context
    
    def _create_document_analysis(
        self,
        job: DocumentJob,
        chunks: List[Dict],
        metadata: Dict[str, Any],
        analysis_text: str,
        framework: ComplianceFramework
    ) -> DocumentAnalysis:
        """Create DocumentAnalysis from processing results"""
        
        from app.models.compliance import DocumentLanguage, GermanComplianceInsights
        
        # Extract German insights if applicable
        german_insights = None
        if job.is_german_compliance:
            # Aggregate German terms from chunks
            all_german_terms = []
            for chunk in chunks:
                chunk_terms = chunk.get('german_terms', {})
                for terms_list in chunk_terms.values():
                    all_german_terms.extend(terms_list)
            
            german_insights = GermanComplianceInsights(
                dsgvo_articles_found=[
                    indicator.replace('gdpr_', '').replace('_', ' ').title()
                    for indicator in job.compliance_indicators
                    if indicator.startswith('gdpr_')
                ],
                german_terms_detected=list(set(all_german_terms)),
                compliance_completeness=0.75,  # Placeholder for now
                german_authority_references=[
                    indicator.replace('authority_', '').upper()
                    for indicator in job.compliance_indicators
                    if indicator.startswith('authority_')
                ]
            )
        
        # Determine document language
        language_map = {
            'de': DocumentLanguage.GERMAN,
            'en': DocumentLanguage.ENGLISH,
            'mixed': DocumentLanguage.MIXED,
            'unknown': DocumentLanguage.UNKNOWN
        }
        
        doc_language = language_map.get(job.language_detected, DocumentLanguage.UNKNOWN)
        
        # Extract basic compliance gaps from analysis
        compliance_gaps = []
        if 'missing' in analysis_text.lower():
            compliance_gaps.append("Missing compliance elements detected")
        if 'insufficient' in analysis_text.lower():
            compliance_gaps.append("Insufficient documentation identified")
        if 'unclear' in analysis_text.lower():
            compliance_gaps.append("Unclear compliance requirements")
        
        # Extract risk indicators
        risk_indicators = []
        if any(term in analysis_text.lower() for term in ['risk', 'violation', 'non-compliant']):
            risk_indicators.append("Compliance risks identified")
        if job.priority == job.priority.CRITICAL:
            risk_indicators.append("High-priority compliance document")
        
        return DocumentAnalysis(
            filename=job.filename,
            document_language=doc_language,
            compliance_summary=analysis_text[:500],  # Truncate for summary
            control_mappings=[],  # Will be enhanced in future
            compliance_gaps=compliance_gaps,
            risk_indicators=risk_indicators,
            german_insights=german_insights,
            original_size=job.size,
            processing_time=job.processing_time or 0.0
        )
    
    def _create_fallback_analysis(
        self,
        job: DocumentJob,
        metadata: Dict[str, Any],
        framework: ComplianceFramework
    ) -> DocumentAnalysis:
        """Create fallback analysis when OpenAI fails"""
        
        from app.models.compliance import DocumentLanguage
        
        language_map = {
            'de': DocumentLanguage.GERMAN,
            'en': DocumentLanguage.ENGLISH,
            'mixed': DocumentLanguage.MIXED,
            'unknown': DocumentLanguage.UNKNOWN
        }
        
        return DocumentAnalysis(
            filename=job.filename,
            document_language=language_map.get(job.language_detected, DocumentLanguage.UNKNOWN),
            compliance_summary=f"Automated analysis completed for {job.filename}. "
                             f"Document contains {len(job.compliance_indicators)} compliance indicators. "
                             f"Manual review recommended for detailed compliance assessment.",
            control_mappings=[],
            compliance_gaps=["Detailed analysis unavailable - manual review needed"],
            risk_indicators=["Analysis service unavailable"],
            german_insights=None,
            original_size=job.size,
            processing_time=job.processing_time or 0.0
        )
    
    def _update_processing_stats(self, results: List[ProcessingResult]):
        """Update global processing statistics"""
        
        for result in results:
            self.processing_stats['total_processed'] += 1
            self.processing_stats['total_time'] += result.processing_time
            
            if not result.success:
                self.processing_stats['errors'] += 1
            
            if result.job.is_german_compliance:
                self.processing_stats['german_docs'] += 1
            
            self.processing_stats['chunks_created'] += result.chunks_created
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get current processing statistics"""
        
        stats = self.processing_stats.copy()
        
        if stats['total_processed'] > 0:
            stats['avg_processing_time'] = stats['total_time'] / stats['total_processed']
            stats['success_rate'] = (stats['total_processed'] - stats['errors']) / stats['total_processed']
            stats['german_doc_percentage'] = stats['german_docs'] / stats['total_processed']
        else:
            stats['avg_processing_time'] = 0.0
            stats['success_rate'] = 0.0
            stats['german_doc_percentage'] = 0.0
        
        return stats