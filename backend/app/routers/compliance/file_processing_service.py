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
from dataclasses import dataclass

# Graceful import of magic with fallback
try:
    import magic
    MAGIC_AVAILABLE = True
except ImportError:
    MAGIC_AVAILABLE = False
    magic = None

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
        
        # Initialize libmagic for MIME type detection with graceful fallback
        self.magic_available = False
        self.magic_mime = None
        
        if MAGIC_AVAILABLE:
            try:
                self.magic_mime = magic.Magic(mime=True)
                self.magic_available = True
                logger.info("✅ libmagic initialized successfully - using accurate MIME detection")
            except Exception as e:
                logger.warning(f"libmagic initialization failed, using filename-based detection: {e}")
                self.magic_available = False
        else:
            logger.info("📋 python-magic not available - using filename-based MIME detection")
        
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
        
        processed_files = []
        total_size = 0
        
        for file in files:
            processed_file = await self._process_single_file(file)
            processed_files.append(processed_file)
            total_size += processed_file.size
            
            # Check total batch size
            if total_size > getattr(settings, 'max_total_file_size_bytes', 50 * 1024 * 1024):
                raise HTTPException(
                    status_code=400,
                    detail=f"Total batch size exceeds limit. Current: {total_size / (1024*1024):.1f}MB"
                )
        
        logger.info(f"Successfully processed {len(processed_files)} files, total size: {total_size / (1024*1024):.2f}MB")
        return processed_files
    
    async def _process_single_file(self, file: UploadFile) -> ProcessedFile:
        """Process and validate a single uploaded file"""
        
        # Read file content
        content = await file.read()
        await file.seek(0)  # Reset file pointer for potential re-reading
        
        # Validate file size
        if len(content) > self.max_file_size:
            raise HTTPException(
                status_code=400,
                detail=f"File {file.filename} exceeds maximum size limit. "
                       f"Size: {len(content) / (1024*1024):.1f}MB, "
                       f"Limit: {self.max_file_size / (1024*1024):.1f}MB"
            )
        
        # Validate file extension
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in self.allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"File type '{file_ext}' not supported. "
                       f"Allowed: {', '.join(self.allowed_extensions)}"
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
        
        # Try libmagic first if available
        if self.magic_available and self.magic_mime:
            try:
                detected_mime = self.magic_mime.from_buffer(content)
                logger.debug(f"MIME type detected via libmagic: {detected_mime}")
                return detected_mime
            except Exception as e:
                logger.warning(f"libmagic MIME detection failed for {filename}: {e}")
        
        # Fallback to filename-based detection
        file_ext = Path(filename).suffix.lower()
        mime_type_map = {
            '.pdf': 'application/pdf',
            '.doc': 'application/msword',
            '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            '.txt': 'text/plain',
            '.md': 'text/markdown',
            '.rtf': 'application/rtf',
            '.odt': 'application/vnd.oasis.opendocument.text',
            '.html': 'text/html',
            '.xml': 'application/xml'
        }
        
        detected_mime = mime_type_map.get(file_ext, 'application/octet-stream')
        logger.debug(f"MIME type detected via filename: {detected_mime}")
        return detected_mime
    
    async def _detect_encoding_and_content(self, content: bytes) -> Tuple[str, bool]:
        """Detect text encoding and determine if file contains readable text"""
        
        # Try to detect encoding
        encoding = "unknown"
        has_text_content = False
        
        # Common encodings to try
        encodings_to_try = ['utf-8', 'utf-16', 'latin1', 'cp1252', 'ascii']
        
        for enc in encodings_to_try:
            try:
                # Try to decode first 1KB to test encoding
                test_content = content[:1024].decode(enc)
                
                # Check if it looks like text (printable characters)
                printable_ratio = sum(1 for c in test_content if c.isprintable() or c.isspace()) / len(test_content)
                
                if printable_ratio > 0.7:  # 70% printable characters
                    encoding = enc
                    has_text_content = True
                    break
                    
            except (UnicodeDecodeError, UnicodeError):
                continue
        
        # If no encoding worked, check if it's binary
        if not has_text_content:
            # Binary files often have lots of null bytes
            null_ratio = content.count(b'\x00') / len(content) if content else 0
            has_text_content = null_ratio < 0.1  # Less than 10% null bytes might still be text
        
        return encoding, has_text_content
    
    async def _calculate_processing_complexity(
        self, 
        content: bytes, 
        filename: str, 
        mime_type: str
    ) -> float:
        """Calculate processing complexity score (0.0 to 1.0)"""
        
        complexity = 0.0
        
        # Base complexity by file size
        size_mb = len(content) / (1024 * 1024)
        size_complexity = min(size_mb / 10.0, 0.4)  # Max 0.4 for size
        complexity += size_complexity
        
        # Complexity by file type
        type_complexity_map = {
            'application/pdf': 0.3,
            'application/msword': 0.2,
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document': 0.2,
            'text/plain': 0.1,
            'text/markdown': 0.1,
            'application/rtf': 0.15,
            'application/vnd.oasis.opendocument.text': 0.2
        }
        complexity += type_complexity_map.get(mime_type, 0.25)
        
        # Additional complexity factors
        if 'pdf' in mime_type.lower():
            complexity += 0.1  # PDFs need more processing
        
        if 'xml' in mime_type.lower() or 'docx' in filename.lower():
            complexity += 0.05  # Structured documents
        
        return min(complexity, 1.0)  # Cap at 1.0
    
    async def _estimate_processing_time(
        self, 
        file_size: int, 
        complexity: float, 
        mime_type: str
    ) -> float:
        """Estimate processing time in seconds"""
        
        # Base time: 1 second per MB
        base_time = file_size / (1024 * 1024)
        
        # Complexity multiplier
        complexity_multiplier = 1.0 + (complexity * 2.0)  # 1x to 3x based on complexity
        
        # Type-specific multipliers
        type_multipliers = {
            'application/pdf': 2.0,  # PDFs take longer
            'application/msword': 1.5,
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document': 1.5,
            'text/plain': 0.5,  # Text is fast
            'text/markdown': 0.5
        }
        
        type_multiplier = type_multipliers.get(mime_type, 1.0)
        
        estimated_time = base_time * complexity_multiplier * type_multiplier
        
        # Minimum 0.5 seconds, maximum 60 seconds
        return max(0.5, min(estimated_time, 60.0))
    
    async def get_batch_processing_summary(self, processed_files: List[ProcessedFile]) -> Dict[str, Any]:
        """Generate summary statistics for a batch of processed files"""
        
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
            },
            "magic_detection_status": {
                "libmagic_available": self.magic_available,
                "detection_method": "libmagic" if self.magic_available else "filename_based"
            }
        }
    
    def convert_to_tuples(self, processed_files: List[ProcessedFile]) -> List[Tuple[str, bytes, int]]:
        """Convert ProcessedFile objects to tuple format for backward compatibility"""
        return [
            (pf.filename, pf.content, pf.size) 
            for pf in processed_files
        ]