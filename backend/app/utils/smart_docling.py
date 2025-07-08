# app/utils/smart_docling.py - Intelligent Docling Management

import os
import time
from typing import Optional, Any, Dict
import structlog

logger = structlog.get_logger()

class SmartDoclingManager:
    """Intelligent Docling loading based on deployment context"""
    
    def __init__(self):
        self._converter = None
        self._initialization_attempted = False
        self._docling_available = None
        self._initialization_time = None
        
        # Environment detection
        self.environment = os.getenv("ENVIRONMENT", "production").lower()
        self.is_development = self.environment == "development"
        self.is_testing = self.environment == "testing"
        self.is_production = self.environment == "production"
        
        # Configuration flags
        self.force_enable = os.getenv("FORCE_DOCLING", "false").lower() == "true"
        self.force_disable = os.getenv("SKIP_DOCLING", "false").lower() == "true"
        self.lazy_load = os.getenv("DOCLING_LAZY_LOAD", "true").lower() == "true"
        self.skip_heavy_models = os.getenv("SKIP_DOCLING_HEAVY_MODELS", "false").lower() == "true"
        
        logger.info(
            "SmartDoclingManager initialized",
            environment=self.environment,
            force_enable=self.force_enable,
            force_disable=self.force_disable,
            lazy_load=self.lazy_load
        )
    
    @property
    def docling_available(self) -> bool:
        """Check if Docling is available (cached check)"""
        if self._docling_available is None:
            self._docling_available = self._check_docling_availability()
        return self._docling_available
    
    def _check_docling_availability(self) -> bool:
        """Check if Docling can be imported"""
        if self.force_disable:
            logger.info("Docling disabled via SKIP_DOCLING environment variable")
            return False
            
        try:
            import docling
            logger.info("Docling import check successful")
            return True
        except ImportError as e:
            logger.warning(f"Docling not available: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error checking Docling: {e}")
            return False
    
    @property
    def converter(self) -> Optional[Any]:
        """Get Docling converter with lazy initialization"""
        if not self.docling_available:
            return None
            
        if self._converter is None and not self._initialization_attempted:
            self._initialization_attempted = True
            start_time = time.time()
            
            try:
                from docling.document_converter import DocumentConverter
                
                # Initialize with environment-appropriate settings
                converter_config = self._get_converter_config()
                self._converter = DocumentConverter(**converter_config)
                
                self._initialization_time = time.time() - start_time
                logger.info(
                    "Docling DocumentConverter initialized",
                    initialization_time=self._initialization_time,
                    environment=self.environment
                )
                
            except Exception as e:
                logger.error(f"Failed to initialize Docling converter: {e}")
                self._converter = None
                
        return self._converter
    
    def _get_converter_config(self) -> Dict[str, Any]:
        """Get environment-appropriate converter configuration"""
        config = {}
        
        # Development optimizations
        if self.is_development:
            config.update({
                "enable_ocr": False,  # Skip OCR for faster processing
                "enable_table_structure_recognition": True,  # Keep table detection
                "enable_picture_extraction": False,  # Skip image extraction
            })
        
        # Production settings
        elif self.is_production:
            config.update({
                "enable_ocr": True,
                "enable_table_structure_recognition": True,
                "enable_picture_extraction": True,
            })
        
        # Testing settings
        else:  # testing environment
            config.update({
                "enable_ocr": False,  # Faster tests
                "enable_table_structure_recognition": True,
                "enable_picture_extraction": False,
            })
        
        return config
    
    def should_use_docling_for_file(self, filename: str, file_size: int = 0) -> bool:
        """Intelligent decision on whether to use Docling for a specific file"""
        if not self.docling_available:
            return False
            
        if self.force_enable:
            return True
            
        if self.force_disable:
            return False
        
        file_ext = filename.lower().split('.')[-1] if '.' in filename else ''
        
        # File type priority
        high_value_formats = ['pdf', 'docx', 'doc']  # Docling adds most value
        medium_value_formats = ['txt', 'md']         # Fallback works well
        
        # Environment-based decisions
        if self.is_development:
            # Development: Only use Docling for high-value formats
            use_docling = file_ext in high_value_formats
            
            # Exception: Use Docling for large text files (might have structure)
            if file_ext in medium_value_formats and file_size > 50000:  # 50KB+
                use_docling = True
                
        elif self.is_testing:
            # Testing: Use Docling for most formats but with optimizations
            use_docling = file_ext in high_value_formats + medium_value_formats
            
        else:  # production
            # Production: Use Docling for all supported formats
            use_docling = file_ext in high_value_formats + medium_value_formats
        
        logger.debug(
            "Docling usage decision",
            filename=filename,
            file_ext=file_ext,
            file_size=file_size,
            environment=self.environment,
            use_docling=use_docling
        )
        
        return use_docling
    
    def get_processing_mode(self, filename: str, file_size: int = 0) -> str:
        """Get recommended processing mode with rationale"""
        if not self.docling_available:
            return "fallback_unavailable"
        
        if self.should_use_docling_for_file(filename, file_size):
            if self.skip_heavy_models:
                return "docling_fast"
            else:
                return "docling_full"
        else:
            return "fallback_optimal"
    
    def get_status(self) -> Dict[str, Any]:
        """Get current Docling manager status"""
        return {
            "environment": self.environment,
            "docling_available": self.docling_available,
            "converter_initialized": self._converter is not None,
            "initialization_time": self._initialization_time,
            "lazy_load": self.lazy_load,
            "force_enable": self.force_enable,
            "force_disable": self.force_disable,
            "skip_heavy_models": self.skip_heavy_models,
            "initialization_attempted": self._initialization_attempted
        }

# Global instance
smart_docling = SmartDoclingManager()

# Convenience functions for easy imports
def get_docling_converter():
    """Get Docling converter if available"""
    return smart_docling.converter

def should_use_docling(filename: str, file_size: int = 0) -> bool:
    """Check if Docling should be used for this file"""
    return smart_docling.should_use_docling_for_file(filename, file_size)

def get_docling_status() -> Dict[str, Any]:
    """Get Docling availability status"""
    return smart_docling.get_status()