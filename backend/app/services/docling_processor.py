# app/services/docling_processor.py - Day 2 Document Intelligence
import hashlib
import tempfile
import os
import asyncio
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
from datetime import datetime, timezone
import structlog

try:
    from docling.document_converter import DocumentConverter
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.document import Document as DoclingDocument
    DOCLING_AVAILABLE = True
except ImportError:
    DOCLING_AVAILABLE = False
    logger = structlog.get_logger()
    logger.warning("Docling not available - using fallback text processing")

from app.utils.german_detection import GermanComplianceDetector
from app.config import settings

logger = structlog.get_logger()

class DoclingProcessor:
    """Enhanced document processing with Docling intelligence for German compliance"""
    
    def __init__(self):
        # FIXED: Check if Docling is available before initializing
        if DOCLING_AVAILABLE:
            try:
                self.converter = DocumentConverter()
            except Exception as e:
                logger.warning(f"Failed to initialize Docling converter: {e}")
                self.converter = None
        else:
            self.converter = None
            
        self.detector = GermanComplianceDetector()
        self.supported_formats = {
            '.pdf': 'pdf',
            '.docx': 'docx',
            '.doc': 'doc',
            '.txt': 'text',
            '.md': 'markdown'
        }
        
    async def process_document(
        self, 
        file_content: bytes, 
        filename: str,
        workspace_id: str,
        user_id: str
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Process document with Docling for intelligent chunking
        Returns: (chunks, metadata)
        """
        
        start_time = datetime.now(timezone.utc)
        file_extension = Path(filename).suffix.lower()
        
        logger.info(
            "Starting document processing",
            filename=filename,
            file_size=len(file_content),
            workspace_id=workspace_id,
            user_id=user_id,
            docling_available=DOCLING_AVAILABLE
        )
        
        # Validate file format
        if file_extension not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {file_extension}")
        
        # Create content hash for deduplication
        content_hash = hashlib.sha256(file_content).hexdigest()
        
        # FIXED: If Docling is not available or is text/markdown, use fallback immediately
        if not DOCLING_AVAILABLE or not self.converter or file_extension in ['.txt', '.md']:
            return await self._process_with_fallback(
                file_content, filename, content_hash, workspace_id
            )
        
        # Create temporary file for Docling processing
        temp_file = None
        temp_path = None
        
        try:
            with tempfile.NamedTemporaryFile(
                suffix=file_extension,
                delete=False
            ) as temp_file:
                temp_file.write(file_content)
                temp_path = temp_file.name
            
            # Convert document with Docling
            docling_result = await self._convert_with_docling(temp_path, filename)
            
            if docling_result:
                # Extract document metadata
                document_metadata = await self._extract_document_metadata(
                    docling_result, filename, content_hash, len(file_content)
                )
                
                # Create intelligent chunks
                chunks = await self._create_intelligent_chunks(
                    docling_result, filename, workspace_id
                )
            else:
                # Docling failed, use fallback
                return await self._process_with_fallback(
                    file_content, filename, content_hash, workspace_id
                )
            
            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            
            logger.info(
                "Document processing completed",
                filename=filename,
                chunk_count=len(chunks),
                processing_time=processing_time,
                german_content_detected=document_metadata.get('german_content_detected', False)
            )
            
            # Add processing metadata
            document_metadata.update({
                'processing_time_seconds': processing_time,
                'chunk_count': len(chunks),
                'docling_version': '1.0.0' if DOCLING_AVAILABLE else 'fallback',
                'processed_at': start_time.isoformat()
            })
            
            return chunks, document_metadata
            
        except Exception as e:
            logger.error(
                "Document processing failed, using fallback",
                filename=filename,
                error=str(e)
            )
            return await self._process_with_fallback(
                file_content, filename, content_hash, workspace_id
            )
            
        finally:
            # Clean up temporary file
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except Exception as e:
                    logger.warning("Failed to cleanup temp file", temp_path=temp_path, error=str(e))
    
    async def _process_with_fallback(
        self,
        file_content: bytes,
        filename: str,
        content_hash: str,
        workspace_id: str
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Process document using fallback text extraction"""
        
        logger.info(
            "Using fallback text processing",
            filename=filename,
            reason="Docling unavailable or text format"
        )
        
        # Try to decode content as text
        try:
            if isinstance(file_content, bytes):
                content = file_content.decode('utf-8')
            else:
                content = str(file_content)
        except UnicodeDecodeError:
            # Try different encodings
            for encoding in ['utf-8', 'latin-1', 'cp1252']:
                try:
                    content = file_content.decode(encoding)
                    break
                except:
                    continue
            else:
                # If all fail, use lossy decoding
                content = file_content.decode('utf-8', errors='replace')
        
        # Create metadata
        metadata = {
            'filename': filename,
            'original_filename': filename,
            'content_hash': content_hash,
            'file_size': len(file_content),
            'file_type': Path(filename).suffix.lower().lstrip('.'),
            'processing_method': 'fallback_text',
            'docling_success': False
        }
        
        # Detect language and German compliance terms
        language, confidence = self.detector.detect_language(content, filename)
        german_terms = self.detector.extract_german_terms(content)
        gdpr_articles = self.detector.extract_gdpr_articles(content)
        
        metadata.update({
            'language_detected': language,
            'language_confidence': confidence,
            'german_content_detected': language == 'de' or bool(german_terms),
            'german_document_type': self._classify_german_document_type(filename, content, german_terms),
            'german_terms_count': sum(len(terms) for terms in german_terms.values()),
            'dsgvo_articles_found': gdpr_articles,
            'compliance_category': self._classify_compliance_category(content, german_terms)
        })
        
        # Create chunks using fallback method
        chunks = await self._fallback_text_chunking(content, filename)
        
        return chunks, metadata
    
    async def _convert_with_docling(
        self, 
        file_path: str, 
        filename: str
    ) -> Optional[Any]:
        """Convert document using Docling with error handling"""
        
        if not self.converter:
            return None
            
        try:
            # Run Docling conversion in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            docling_result = await loop.run_in_executor(
                None, 
                self.converter.convert, 
                file_path
            )
            
            # Check if conversion was successful
            if hasattr(docling_result, 'document'):
                return docling_result.document
            else:
                logger.warning(
                    "Docling conversion returned unexpected result",
                    filename=filename,
                    result_type=type(docling_result)
                )
                return None
                
        except Exception as e:
            logger.error(
                "Docling conversion failed",
                filename=filename,
                error=str(e)
            )
            return None
    
    async def _extract_document_metadata(
        self,
        docling_result: Any,
        filename: str,
        content_hash: str,
        file_size: int
    ) -> Dict[str, Any]:
        """Extract comprehensive document metadata"""
        
        # Basic metadata
        metadata = {
            'filename': filename,
            'original_filename': filename,
            'content_hash': content_hash,
            'file_size': file_size,
            'file_type': Path(filename).suffix.lower().lstrip('.'),
            'processing_method': 'docling',
            'docling_success': True
        }
        
        try:
            # Extract content for language detection
            if hasattr(docling_result, 'export_to_markdown'):
                content = docling_result.export_to_markdown()
            else:
                content = str(docling_result)
            
            # Detect language and German compliance terms
            language, confidence = self.detector.detect_language(content, filename)
            german_terms = self.detector.extract_german_terms(content)
            gdpr_articles = self.detector.extract_gdpr_articles(content)
            
            # Determine German document type
            german_doc_type = self._classify_german_document_type(
                filename, content, german_terms
            )
            
            # Extract Docling-specific metadata
            docling_metadata = {
                'page_count': getattr(docling_result, 'page_count', 1),
                'has_tables': self._detect_tables(docling_result),
                'has_images': False,  # Disabled for now
                'document_structure': self._analyze_document_structure(docling_result)
            }
            
            metadata.update({
                'language_detected': language,
                'language_confidence': confidence,
                'german_content_detected': language == 'de' or bool(german_terms),
                'german_document_type': german_doc_type,
                'german_terms_count': sum(len(terms) for terms in german_terms.values()),
                'dsgvo_articles_found': gdpr_articles,
                'compliance_category': self._classify_compliance_category(content, german_terms),
                'docling_metadata': docling_metadata
            })
            
        except Exception as e:
            logger.warning(
                "Metadata extraction partially failed",
                filename=filename,
                error=str(e)
            )
            metadata['extraction_error'] = str(e)
        
        return metadata
    
    async def _create_intelligent_chunks(
        self,
        docling_result: Any,
        filename: str,
        workspace_id: str
    ) -> List[Dict[str, Any]]:
        """Create semantically meaningful chunks from Docling output"""
        
        chunks = []
        
        try:
            # Get document structure from Docling
            if hasattr(docling_result, 'body') and hasattr(docling_result.body, 'elements'):
                chunks = await self._process_docling_elements(
                    docling_result.body.elements, filename
                )
            else:
                # Fallback to text-based chunking
                if hasattr(docling_result, 'export_to_markdown'):
                    content = docling_result.export_to_markdown()
                else:
                    content = str(docling_result)
                chunks = await self._fallback_text_chunking(content, filename)
                
        except Exception as e:
            logger.warning(
                "Intelligent chunking failed, using fallback",
                filename=filename,
                error=str(e)
            )
            # Fallback to simple text chunking
            try:
                content = str(docling_result)
                chunks = await self._fallback_text_chunking(content, filename)
            except Exception as fallback_error:
                logger.error(
                    "Fallback chunking also failed",
                    filename=filename,
                    error=str(fallback_error)
                )
                # Return at least one chunk with the content we have
                chunks = [{
                    'chunk_index': 0,
                    'content': str(docling_result)[:1000],  # First 1000 chars
                    'chunk_type': 'error_fallback',
                    'page_number': 1,
                    'error': True
                }]
        
        # Limit chunks for Day 2 performance
        if len(chunks) > settings.max_chunks_per_document:
            logger.info(
                "Limiting chunks for performance",
                filename=filename,
                original_count=len(chunks),
                limited_count=settings.max_chunks_per_document
            )
            chunks = chunks[:settings.max_chunks_per_document]
        
        return chunks
    
    async def _process_docling_elements(
        self,
        elements: List[Any],
        filename: str
    ) -> List[Dict[str, Any]]:
        """Process Docling document elements into intelligent chunks"""
        
        chunks = []
        chunk_index = 0
        
        for element in elements:
            try:
                chunk_content = self._extract_element_text(element)
                
                if len(chunk_content.strip()) < 20:  # Skip very short chunks
                    continue
                
                # Detect German terms in this chunk
                language, confidence = self.detector.detect_language(chunk_content, filename)
                german_terms = self.detector.extract_german_terms(chunk_content)
                gdpr_articles = self.detector.extract_gdpr_articles(chunk_content)
                
                # Generate compliance tags for this chunk
                compliance_tags = self._generate_compliance_tags(
                    chunk_content, german_terms, gdpr_articles
                )
                
                # Calculate structural importance
                structural_importance = self._calculate_structural_importance(
                    element, chunk_content
                )
                
                chunk = {
                    'chunk_index': chunk_index,
                    'content': chunk_content,
                    'content_hash': hashlib.sha256(chunk_content.encode()).hexdigest()[:16],
                    'chunk_type': self._classify_element_type(element),
                    'page_number': getattr(element, 'page', 1),
                    'position_in_page': getattr(element, 'position', chunk_index),
                    'char_count': len(chunk_content),
                    'confidence_score': confidence,
                    'structural_importance': structural_importance,
                    'language_detected': language,
                    'german_terms': german_terms,
                    'dsgvo_articles': gdpr_articles,
                    'compliance_tags': compliance_tags
                }
                
                chunks.append(chunk)
                chunk_index += 1
                
            except Exception as e:
                logger.warning(
                    "Failed to process document element",
                    filename=filename,
                    element_index=chunk_index,
                    error=str(e)
                )
                continue
        
        return chunks
    
    def _extract_element_text(self, element) -> str:
        """Extract text content from Docling element"""
        try:
            # Try various attributes that might contain text
            if hasattr(element, 'text'):
                return str(element.text)
            elif hasattr(element, 'content'):
                return str(element.content)
            elif hasattr(element, 'value'):
                return str(element.value)
            elif hasattr(element, 'get_text'):
                return element.get_text()
            else:
                # Last resort - convert to string
                return str(element)
        except Exception:
            return ""
    
    def _classify_element_type(self, element) -> str:
        """Classify Docling element type for German compliance documents"""
        element_type_name = str(type(element).__name__).lower()
        element_str = str(element).lower()
        
        # German compliance document structure detection
        if any(term in element_str for term in ['datenschutz', 'dsgvo', 'artikel', 'rechtsgr']):
            if 'table' in element_type_name or 'tabelle' in element_str:
                return 'compliance_table'
            elif any(term in element_str for term in ['§', 'artikel', 'art.']):
                return 'legal_reference'
            else:
                return 'compliance_text'
        
        # Standard document structure
        if 'table' in element_type_name or 'tabular' in element_type_name:
            return 'table'
        elif any(term in element_type_name for term in ['heading', 'title', 'header']):
            return 'header'
        elif 'list' in element_type_name:
            return 'list'
        elif 'paragraph' in element_type_name:
            return 'paragraph'
        elif 'figure' in element_type_name or 'image' in element_type_name:
            return 'figure'
        else:
            return 'text'
    
    def _calculate_structural_importance(self, element, content: str) -> float:
        """Calculate how structurally important this chunk is"""
        importance = 0.5  # Base importance
        
        element_type = self._classify_element_type(element)
        content_lower = content.lower()
        
        # Type-based importance
        type_weights = {
            'header': 0.9,
            'legal_reference': 0.95,
            'compliance_table': 0.8,
            'compliance_text': 0.7,
            'table': 0.6,
            'list': 0.5,
            'paragraph': 0.4,
            'text': 0.3
        }
        
        importance = type_weights.get(element_type, 0.5)
        
        # German legal content boosts importance
        german_legal_indicators = [
            'dsgvo', 'datenschutz', 'artikel', 'rechtsgrundlage',
            'verarbeitung', 'betroffenenrechte', 'aufsichtsbehörde'
        ]
        
        legal_matches = sum(1 for term in german_legal_indicators if term in content_lower)
        if legal_matches > 0:
            importance = min(1.0, importance + (legal_matches * 0.1))
        
        return round(importance, 2)
    
    def _generate_compliance_tags(
        self,
        content: str,
        german_terms: Dict[str, List[str]],
        gdpr_articles: List[str]
    ) -> List[str]:
        """Generate compliance-relevant tags for chunk"""
        
        tags = []
        content_lower = content.lower()
        
        # GDPR/DSGVO article tags
        if gdpr_articles:
            tags.extend([
                f"gdpr_{article.lower().replace('.', '_').replace(' ', '_')}" 
                for article in gdpr_articles
            ])
        
        # German compliance term tags
        if german_terms:
            for category, terms in german_terms.items():
                if terms:
                    tags.append(f"german_{category}")
        
        # Content-based compliance tags
        compliance_indicators = {
            'personal_data': [
                'personal data', 'personenbezogene daten', 'customer data',
                'kundendaten', 'mitarbeiterdaten', 'nutzerdaten'
            ],
            'consent': [
                'consent', 'einwilligung', 'opt-in', 'permission',
                'zustimmung', 'genehmigung'
            ],
            'privacy_policy': [
                'privacy policy', 'datenschutzerklärung', 'privacy notice',
                'datenschutzrichtlinie', 'datenschutzhinweise'
            ],
            'security': [
                'security', 'sicherheit', 'encryption', 'verschlüsselung',
                'schutzmaßnahmen', 'technische maßnahmen'
            ],
            'retention': [
                'retention', 'aufbewahrung', 'delete', 'löschen',
                'speicherdauer', 'löschfristen'
            ],
            'breach': [
                'breach', 'verletzung', 'incident', 'vorfall',
                'datenpanne', 'sicherheitsvorfall'
            ],
            'rights': [
                'rights', 'rechte', 'access', 'zugang', 'rectification',
                'auskunft', 'berichtigung', 'löschung'
            ],
            'transfer': [
                'transfer', 'übermittlung', 'third party', 'dritte',
                'weitergabe', 'übertragung'
            ],
            'dpo': [
                'dpo', 'data protection officer', 'datenschutzbeauftragter',
                'dsb', 'datenschutzverantwortlicher'
            ],
            'authority': [
                'supervisory authority', 'aufsichtsbehörde', 'bfdi',
                'landesdatenschutz', 'datenschutzbehörde'
            ]
        }
        
        for tag, indicators in compliance_indicators.items():
            if any(indicator in content_lower for indicator in indicators):
                tags.append(f"content_{tag}")
        
        # German industry-specific tags
        industry_terms = {
            'automotive': ['fahrzeug', 'automotive', 'kfz', 'auto'],
            'healthcare': ['gesundheit', 'patient', 'medical', 'kranken'],
            'manufacturing': ['produktion', 'fertigung', 'herstellung', 'fabrik'],
            'financial': ['bank', 'finanz', 'kredit', 'versicherung']
        }
        
        for industry, terms in industry_terms.items():
            if any(term in content_lower for term in terms):
                tags.append(f"industry_{industry}")
        
        return list(set(tags))  # Remove duplicates
    
    async def _fallback_text_chunking(
        self, 
        content: str, 
        filename: str
    ) -> List[Dict[str, Any]]:
        """Fallback to simple text chunking when Docling fails"""
        
        chunks = []
        chunk_size = settings.chunk_size
        overlap = settings.chunk_overlap
        
        # Simple paragraph-based chunking
        paragraphs = content.split('\n\n')
        current_chunk = ""
        chunk_index = 0
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
                
            if len(current_chunk) + len(paragraph) > chunk_size and current_chunk:
                # Process current chunk
                chunk = await self._create_fallback_chunk(
                    current_chunk, chunk_index, filename
                )
                chunks.append(chunk)
                chunk_index += 1
                
                # Start new chunk with overlap
                words = current_chunk.split()
                overlap_words = words[-overlap:] if len(words) > overlap else words
                current_chunk = ' '.join(overlap_words) + '\n\n' + paragraph
            else:
                current_chunk += '\n\n' + paragraph if current_chunk else paragraph
        
        # Add final chunk
        if current_chunk.strip():
            chunk = await self._create_fallback_chunk(
                current_chunk, chunk_index, filename
            )
            chunks.append(chunk)
        
        # If no chunks were created, create at least one
        if not chunks and content.strip():
            chunks.append(await self._create_fallback_chunk(
                content[:chunk_size], 0, filename
            ))
        
        return chunks
    
    async def _create_fallback_chunk(
        self, 
        content: str, 
        chunk_index: int, 
        filename: str
    ) -> Dict[str, Any]:
        """Create a chunk using fallback processing"""
        
        # Detect German terms and language
        language, confidence = self.detector.detect_language(content, filename)
        german_terms = self.detector.extract_german_terms(content)
        gdpr_articles = self.detector.extract_gdpr_articles(content)
        
        # Generate compliance tags
        compliance_tags = self._generate_compliance_tags(
            content, german_terms, gdpr_articles
        )
        
        return {
            'chunk_index': chunk_index,
            'content': content.strip(),
            'content_hash': hashlib.sha256(content.encode()).hexdigest()[:16],
            'chunk_type': 'paragraph',
            'page_number': 1,
            'position_in_page': chunk_index,
            'char_count': len(content),
            'confidence_score': confidence,
            'structural_importance': 0.5,  # Default for fallback
            'language_detected': language,
            'german_terms': german_terms,
            'dsgvo_articles': gdpr_articles,
            'compliance_tags': compliance_tags
        }
    
    def _classify_german_document_type(
        self, 
        filename: str, 
        content: str, 
        german_terms: Dict[str, List[str]]
    ) -> Optional[str]:
        """Classify German compliance document type"""
        
        filename_lower = filename.lower()
        content_lower = content.lower()
        
        # Filename-based classification
        filename_patterns = {
            'datenschutzerklärung': ['datenschutz', 'privacy'],
            'verfahrensverzeichnis': ['verfahren', 'ropa', 'processing'],
            'dsfa': ['dsfa', 'dpia', 'folgenabschätzung'],
            'richtlinie': ['richtlinie', 'policy', 'guideline'],
            'schulung': ['schulung', 'training', 'ausbildung'],
            'vertrag': ['vertrag', 'contract', 'agreement'],
            'einwilligung': ['einwilligung', 'consent', 'zustimmung']
        }
        
        for doc_type, patterns in filename_patterns.items():
            if any(pattern in filename_lower for pattern in patterns):
                return doc_type
        
        # Content-based classification
        content_patterns = {
            'datenschutzerklärung': [
                'diese datenschutzerklärung', 'privacy policy', 'data protection notice'
            ],
            'verfahrensverzeichnis': [
                'verfahrensverzeichnis', 'records of processing', 'ropa'
            ],
            'dsfa': [
                'datenschutz-folgenabschätzung', 'data protection impact assessment'
            ],
            'mitarbeiterschulung': [
                'schulungsplan', 'training material', 'awareness'
            ]
        }
        
        for doc_type, patterns in content_patterns.items():
            if any(pattern in content_lower for pattern in patterns):
                return doc_type
        
        # German terms-based classification
        if german_terms:
            if any('dsgvo' in str(terms).lower() for terms in german_terms.values()):
                return 'dsgvo_dokument'
            if any('verfahren' in str(terms).lower() for terms in german_terms.values()):
                return 'verfahrensdokumentation'
        
        return None
    
    def _classify_compliance_category(
        self, 
        content: str, 
        german_terms: Dict[str, List[str]]
    ) -> str:
        """Classify compliance category of document"""
        
        content_lower = content.lower()
        
        # Policy documents
        if any(term in content_lower for term in [
            'policy', 'richtlinie', 'datenschutzerklärung', 'privacy notice'
        ]):
            return 'policy'
        
        # Procedures and processes
        if any(term in content_lower for term in [
            'procedure', 'verfahren', 'process', 'workflow', 'ablauf'
        ]):
            return 'procedure'
        
        # Assessments
        if any(term in content_lower for term in [
            'assessment', 'bewertung', 'analyse', 'prüfung', 'dsfa', 'dpia'
        ]):
            return 'assessment'
        
        # Training materials
        if any(term in content_lower for term in [
            'training', 'schulung', 'awareness', 'sensibilisierung'
        ]):
            return 'training'
        
        # Contracts and agreements
        if any(term in content_lower for term in [
            'contract', 'vertrag', 'agreement', 'vereinbarung'
        ]):
            return 'contract'
        
        # Documentation
        if any(term in content_lower for term in [
            'documentation', 'dokumentation', 'records', 'aufzeichnungen'
        ]):
            return 'documentation'
        
        return 'general'
    
    def _detect_tables(self, docling_result) -> bool:
        """Detect if document contains tables"""
        try:
            if hasattr(docling_result, 'body') and hasattr(docling_result.body, 'elements'):
                return any('table' in str(type(element).__name__).lower() 
                         for element in docling_result.body.elements)
        except Exception:
            pass
        return False
    
    def _analyze_document_structure(self, docling_result) -> Dict[str, int]:
        """Analyze document structure for metadata"""
        structure = {
            'headers': 0,
            'paragraphs': 0,
            'tables': 0,
            'lists': 0,
            'figures': 0
        }
        
        try:
            if hasattr(docling_result, 'body') and hasattr(docling_result.body, 'elements'):
                for element in docling_result.body.elements:
                    element_type = str(type(element).__name__).lower()
                    
                    if any(term in element_type for term in ['heading', 'title', 'header']):
                        structure['headers'] += 1
                    elif 'table' in element_type:
                        structure['tables'] += 1
                    elif 'list' in element_type:
                        structure['lists'] += 1
                    elif any(term in element_type for term in ['figure', 'image']):
                        structure['figures'] += 1
                    elif 'paragraph' in element_type:
                        structure['paragraphs'] += 1
        except Exception:
            pass
        
        return structure