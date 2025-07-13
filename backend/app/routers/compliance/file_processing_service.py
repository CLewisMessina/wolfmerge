# app/routers/compliance/file_processing_service.py
"""
File Processing Service

Handles file validation, preprocessing, content sanitization, and metadata extraction.
This service isolates all file-related operations and provides a clean interface
for the orchestrator to work with validated, processed files.
"""

from typing import List, Tuple, Dict, Any, Optional
from fastapi import UploadFile, HTTPException
import structlog
import asyncio
from pathlib import Path
import hashlib
import magic
from dataclasses import dataclass

from app.config import settings
from app.utils.smart_docling import should_use_docling

logger = structlog.get_logger()

@dataclass 
class FileMetadata:
    """Metadata extracted from processed files"""
    filename: str
    original_size: int
    processed_size: int
    mime_type: str
    file_hash: str
    encoding: str
    has_text_content: bool
    docling_recommended: bool
    processing_complexity: float
    estimated_processing_time: float
    security_flags: List[str]

@dataclass
class ProcessedFile:
    """Container for processed file data and metadata"""
    filename: str
    content: bytes
    size: int
    metadata: FileMetadata
    
    def __post_init__(self):
        """Validate processed file integrity"""
        if len(self.content) != self.size:
            raise ValueError(f"Content size mismatch for {self.filename}")
        
        if self.size != self.metadata.processed_size:
            raise ValueError(f"Metadata size mismatch for {self.filename}")

class FileProcessingService:
    """
    Service for comprehensive file processing including validation,
    sanitization, metadata extraction, and preprocessing.
    
    This service handles all file-related operations before compliance analysis:
    - File type validation and security checks
    - Content sanitization and encoding detection
    - Metadata extraction and complexity analysis
    - Docling integration recommendations
    - File size and batch optimization
    """
    
    def __init__(self):
        self.max_file_size = getattr(settings, 'max_file_size', 10 * 1024 * 1024)  # 10MB default
        self.max_batch_size = getattr(settings, 'max_files_per_batch', 20)
        self.allowed_extensions = getattr(settings, 'allowed_extensions', [
            '.pdf', '.docx', '.doc', '.txt', '.md', '.rtf', '.odt'
        ])
        
        # Initialize libmagic for MIME type detection
        try:
            self.magic_mime = magic.Magic(mime=True)
            self.magic_available = True
        except Exception as e:
            logger.warning(f"libmagic not available, using filename-based detection: {e}")
            self.magic_available = False
        
        # Security patterns to detect in content
        self.security_patterns = {
            'potential_malware': [b'\x4d\x5a', b'\x50\x4b'],  # PE/ZIP headers
            'suspicious_scripts': [b'<script', b'javascript:', b'vbscript:'],
            'embedded_executables': [b'.exe', b'.bat', b'.cmd', b'.scr'],
            'macro_indicators': [b'macros', b'vba', b'Sub ', b'Function ']
        }
    
    async def process_and_validate_files(self, files: List[UploadFile]) -> List[ProcessedFile]:
        """
        Main entry point for file processing and validation.
        
        This method performs comprehensive file processing including:
        - Batch size validation
        - Individual file validation
        - Content processing and sanitization  
        - Metadata extraction
        - Security scanning
        
        Returns a list of ProcessedFile objects ready for compliance analysis.
        """
        
        # Validate batch size
        if len(files) > self.max_batch_size:
            raise HTTPException(
                status_code=400,
                detail=f"Maximum {self.max_batch_size} files allowed per batch. Got {len(files)} files."
            )
        
        if not files:
            raise HTTPException(
                status_code=400,
                detail="No files provided for analysis"
            )
        
        processed_files = []
        total_size = 0
        
        logger.info(f"Starting file processing for {len(files)} files")
        
        # Process files concurrently for better performance
        tasks = []
        for file in files:
            task = self._process_single_file(file)
            tasks.append(task)
        
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"File processing failed for {files[i].filename}: {result}")
                    raise HTTPException(
                        status_code=400,
                        detail=f"File processing failed for {files[i].filename}: {str(result)}"
                    )
                
                processed_file = result
                processed_files.append(processed_file)
                total_size += processed_file.size
                
        except Exception as e:
            if isinstance(e, HTTPException):
                raise
            logger.error(f"Batch file processing failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Batch file processing failed: {str(e)}"
            )
        
        # Validate total batch size
        max_total_size = self.max_file_size * len(files)
        if total_size > max_total_size:
            raise HTTPException(
                status_code=400,
                detail=f"Total batch size ({total_size:,} bytes) exceeds maximum ({max_total_size:,} bytes)"
            )
        
        logger.info(
            f"File processing completed successfully",
            file_count=len(processed_files),
            total_size_mb=total_size / (1024 * 1024),
            avg_complexity=sum(f.metadata.processing_complexity for f in processed_files) / len(processed_files)
        )
        
        return processed_files
    
    async def _process_single_file(self, file: UploadFile) -> ProcessedFile:
        """Process and validate a single uploaded file"""
        
        # Basic filename validation
        if not file.filename:
            raise ValueError("File must have a filename")
        
        # Read file content
        try:
            content = await file.read()
        except Exception as e:
            raise ValueError(f"Failed to read file content: {str(e)}")
        
        # Validate file size
        if len(content) == 0:
            raise ValueError("File is empty")
        
        if len(content) > self.max_file_size:
            raise ValueError(
                f"File size ({len(content):,} bytes) exceeds maximum ({self.max_file_size:,} bytes)"
            )
        
        # Validate file extension
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in self.allowed_extensions:
            raise ValueError(
                f"File type '{file_ext}' not supported. Allowed: {', '.join(self.allowed_extensions)}"
            )
        
        # Perform security scanning
        security_flags = await self._scan_for_security_issues(content, file.filename)
        if security_flags:
            logger.warning(f"Security flags detected in {file.filename}: {security_flags}")
        
        # Extract metadata
        metadata = await self._extract_file_metadata(
            file.filename, content, security_flags
        )
        
        # Create processed file object
        processed_file = ProcessedFile(
            filename=file.filename,
            content=content,
            size=len(content),
            metadata=metadata
        )
        
        logger.debug(
            f"File processed successfully",
            filename=file.filename,
            size=len(content),
            complexity=metadata.processing_complexity,
            docling_recommended=metadata.docling_recommended
        )
        
        return processed_file
    
    async def _scan_for_security_issues(self, content: bytes, filename: str) -> List[str]:
        """Scan file content for potential security issues"""
        
        security_flags = []
        
        try:
            # Check for suspicious binary patterns
            for flag_type, patterns in self.security_patterns.items():
                for pattern in patterns:
                    if pattern in content:
                        security_flags.append(f"{flag_type}_{pattern.hex()[:8]}")
                        break  # Only flag once per category
            
            # Check for excessively large files (potential DoS)
            if len(content) > self.max_file_size * 0.8:  # 80% of max size
                security_flags.append("large_file_warning")
            
            # Check for suspicious filename patterns
            suspicious_filename_patterns = [
                '.exe', '.bat', '.cmd', '.scr', '.vbs', '.js', '.jar'
            ]
            filename_lower = filename.lower()
            for pattern in suspicious_filename_patterns:
                if pattern in filename_lower:
                    security_flags.append(f"suspicious_filename_{pattern}")
            
            # Check for embedded nulls (potential binary injection)
            if b'\x00' in content[:1024]:  # Check first 1KB
                security_flags.append("null_bytes_detected")
            
        except Exception as e:
            logger.warning(f"Security scan failed for {filename}: {e}")
            security_flags.append("security_scan_failed")
        
        return security_flags
    
    async def _extract_file_metadata(
        self, 
        filename: str, 
        content: bytes, 
        security_flags: List[str]
    ) -> FileMetadata:
        """Extract comprehensive metadata from file"""
        
        # Calculate file hash for integrity
        file_hash = hashlib.sha256(content).hexdigest()
        
        # Detect MIME type
        mime_type = await self._detect_mime_type(content, filename)
        
        # Detect encoding and text content
        encoding, has_text_content = await self._detect_encoding_and_content(content)
        
        # Calculate processing complexity
        processing_complexity = await self._calculate_processing_complexity(
            content, filename, mime_type
        )
        
        # Estimate processing time
        estimated_processing_time = await self._estimate_processing_time(
            len(content), processing_complexity, mime_type
        )
        
        # Check if Docling is recommended
        docling_recommended = should_use_docling(filename, len(content))
        
        return FileMetadata(
            filename=filename,
            original_size=len(content),
            processed_size=len(content),
            mime_type=mime_type,
            file_hash=file_hash,
            encoding=encoding,
            has_text_content=has_text_content,
            docling_recommended=docling_recommended,
            processing_complexity=processing_complexity,
            estimated_processing_time=estimated_processing_time,
            security_flags=security_flags
        )
    
    async def _detect_mime_type(self, content: bytes, filename: str) -> str:
        """Detect MIME type using libmagic or filename fallback"""
        
        if self.magic_available:
            try:
                return self.magic_mime.from_buffer(content)
            except Exception as e:
                logger.warning(f"libmagic MIME detection failed: {e}")
        
        # Fallback to filename-based detection
        file_ext = Path(filename).suffix.lower()
        mime_type_map = {
            '.pdf': 'application/pdf',
            '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            '.doc': 'application/msword',
            '.txt': 'text/plain',
            '.md': 'text/markdown',
            '.rtf': 'application/rtf',
            '.odt': 'application/vnd.oasis.opendocument.text'
        }
        
        return mime_type_map.get(file_ext, 'application/octet-stream')
    
    async def _detect_encoding_and_content(self, content: bytes) -> Tuple[str, bool]:
        """Detect text encoding and whether file contains readable text"""
        
        # Try to decode as text to determine if it has text content
        encodings_to_try = ['utf-8', 'utf-16', 'latin-1', 'cp1252']
        
        for encoding in encodings_to_try:
            try:
                decoded = content.decode(encoding)
                # Check if decoded content looks like readable text
                printable_ratio = sum(c.isprintable() or c.isspace() for c in decoded) / len(decoded)
                if printable_ratio > 0.7:  # 70% printable characters
                    return encoding, True
            except (UnicodeDecodeError, UnicodeError):
                continue
        
        # If no encoding works, it's likely binary
        return 'binary', False
    
    async def _calculate_processing_complexity(
        self, 
        content: bytes, 
        filename: str, 
        mime_type: str
    ) -> float:
        """Calculate processing complexity score (0.0 to 3.0)"""
        
        complexity = 0.0
        
        # Base complexity from file size
        size_mb = len(content) / (1024 * 1024)
        complexity += min(1.0, size_mb * 0.2)  # Up to 1.0 for size
        
        # MIME type complexity
        if mime_type in ['application/pdf', 'application/msword']:
            complexity += 0.8  # Complex structured documents
        elif 'officedocument' in mime_type:
            complexity += 0.6  # Modern Office documents
        elif mime_type.startswith('text/'):
            complexity += 0.2  # Simple text files
        else:
            complexity += 0.4  # Unknown/other types
        
        # Content complexity analysis
        try:
            if mime_type.startswith('text/') or 'text' in mime_type:
                text = content.decode('utf-8', errors='ignore')
                
                # Word count complexity
                word_count = len(text.split())
                complexity += min(0.5, word_count / 20000)  # Up to 0.5 for word count
                
                # Structure complexity (tables, lists, formatting)
                structure_indicators = (
                    text.count('\n') + text.count('\t') + 
                    text.count('|') + text.count('*') + text.count('#')
                )
                complexity += min(0.3, structure_indicators / 500)
                
                # Legal/technical complexity
                technical_terms = [
                    'article', 'artikel', 'section', 'clause', 'gdpr', 'dsgvo',
                    'compliance', 'audit', 'regulation', 'verordnung'
                ]
                technical_count = sum(text.lower().count(term) for term in technical_terms)
                complexity += min(0.4, technical_count / 50)
                
        except Exception:
            # If content analysis fails, add moderate complexity
            complexity += 0.3
        
        return min(3.0, complexity)  # Cap at maximum complexity
    
    async def _estimate_processing_time(
        self, 
        file_size: int, 
        complexity: float, 
        mime_type: str
    ) -> float:
        """Estimate processing time in seconds"""
        
        # Base time calculation
        base_time = (file_size / (1024 * 1024)) * 0.5  # 0.5s per MB
        
        # Complexity multiplier
        complexity_multiplier = 1.0 + (complexity / 3.0)  # 1.0 to 2.0x
        
        # MIME type multiplier
        if mime_type == 'application/pdf':
            mime_multiplier = 1.5  # PDFs take longer
        elif 'officedocument' in mime_type:
            mime_multiplier = 1.3  # Office docs moderately complex
        elif mime_type.startswith('text/'):
            mime_multiplier = 0.8  # Text files are faster
        else:
            mime_multiplier = 1.0
        
        estimated_time = base_time * complexity_multiplier * mime_multiplier
        
        # Minimum and maximum bounds
        return max(0.5, min(30.0, estimated_time))  # Between 0.5s and 30s
    
    def get_batch_statistics(self, processed_files: List[ProcessedFile]) -> Dict[str, Any]:
        """Get statistics for the processed file batch"""
        
        if not processed_files:
            return {"error": "No files processed"}
        
        total_size = sum(f.size for f in processed_files)
        avg_complexity = sum(f.metadata.processing_complexity for f in processed_files) / len(processed_files)
        total_estimated_time = sum(f.metadata.estimated_processing_time for f in processed_files)
        
        # File type distribution
        mime_types = {}
        for f in processed_files:
            mime_type = f.metadata.mime_type
            mime_types[mime_type] = mime_types.get(mime_type, 0) + 1
        
        # Security analysis
        total_security_flags = sum(len(f.metadata.security_flags) for f in processed_files)
        files_with_security_flags = sum(1 for f in processed_files if f.metadata.security_flags)
        
        # Docling recommendations
        docling_recommended_count = sum(1 for f in processed_files if f.metadata.docling_recommended)
        
        return {
            "file_count": len(processed_files),
            "total_size_bytes": total_size,
            "total_size_mb": total_size / (1024 * 1024),
            "average_complexity": avg_complexity,
            "total_estimated_processing_time": total_estimated_time,
            "mime_type_distribution": mime_types,
            "security_analysis": {
                "total_flags": total_security_flags,
                "files_with_flags": files_with_security_flags,
                "security_percentage": (files_with_security_flags / len(processed_files)) * 100
            },
            "docling_analysis": {
                "recommended_count": docling_recommended_count,
                "recommendation_percentage": (docling_recommended_count / len(processed_files)) * 100
            },
            "text_content_analysis": {
                "text_files": sum(1 for f in processed_files if f.metadata.has_text_content),
                "binary_files": sum(1 for f in processed_files if not f.metadata.has_text_content)
            }
        }
    
    def convert_to_tuples(self, processed_files: List[ProcessedFile]) -> List[Tuple[str, bytes, int]]:
        """Convert ProcessedFile objects to tuple format for backward compatibility"""
        return [
            (pf.filename, pf.content, pf.size) 
            for pf in processed_files
        ]