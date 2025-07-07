# app/services/file_processor.py
import aiofiles
from typing import List, Tuple
from fastapi import UploadFile, HTTPException

class FileProcessor:
    """Handle file validation and content extraction"""
    
    ALLOWED_MIME_TYPES = {
        'text/plain': ['.txt'],
        'text/markdown': ['.md'],
        'text/x-markdown': ['.md']
    }
    
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    
    @classmethod
    async def validate_and_extract(cls, files: List[UploadFile]) -> List[Tuple[str, str, int]]:
        """
        Validate files and extract content
        Returns: List of (filename, content, size) tuples
        """
        if len(files) > 5:
            raise HTTPException(
                status_code=400,
                detail="Maximum 5 files allowed for Day 1 MVP"
            )
        
        extracted_files = []
        
        for file in files:
            # Validate file size
            if file.size and file.size > cls.MAX_FILE_SIZE:
                raise HTTPException(
                    status_code=400,
                    detail=f"File {file.filename} exceeds maximum size of 10MB"
                )
            
            # Validate file extension
            if not any(file.filename.lower().endswith(ext) 
                      for ext_list in cls.ALLOWED_MIME_TYPES.values() 
                      for ext in ext_list):
                raise HTTPException(
                    status_code=400,
                    detail=f"File type not supported: {file.filename}. Only .txt and .md files allowed in Day 1."
                )
            
            # Extract content
            try:
                content = await file.read()
                text_content = content.decode('utf-8')
                
                # Basic validation
                if len(text_content.strip()) < 10:
                    raise HTTPException(
                        status_code=400,
                        detail=f"File {file.filename} appears to be empty or too short"
                    )
                
                extracted_files.append((file.filename, text_content, len(text_content)))
                
            except UnicodeDecodeError:
                raise HTTPException(
                    status_code=400,
                    detail=f"File {file.filename} contains invalid UTF-8 content"
                )
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Error processing file {file.filename}: {str(e)}"
                )
        
        return extracted_files
    
    @classmethod
    async def secure_cleanup(cls, file_content: str) -> None:
        """Secure cleanup for GDPR compliance (Day 2 will enhance this)"""
        # For Day 1, simple cleanup
        # Day 2 will add proper secure memory clearing
        file_content = None