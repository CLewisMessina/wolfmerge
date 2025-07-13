# app/routers/compliance/authority_detection_service.py
"""
Authority Detection Service

Handles German authority detection, context management, and Big 4 authority engine
integration. This service isolates the authority context scope issues and provides
a clean interface for authority-related functionality.
"""

from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
import structlog
import asyncio
from enum import Enum

from app.services.parallel_processing import UIContextLayer
from app.services.german_authority_engine.integration.authority_endpoints import (
    Big4AuthorityEndpoints, create_big4_authority_endpoints
)

logger = structlog.get_logger()

class AuthorityType(str, Enum):
    """German authority types for classification"""
    BFDI = "bfdi"
    BAYLDA = "baylda" 
    LFD_BW = "lfd_bw"
    LFD_NDS = "lfd_nds"
    UNKNOWN = "unknown"

class IndustryType(str, Enum):
    """Industry types for authority specialization"""
    AUTOMOTIVE = "automotive"
    HEALTHCARE = "healthcare"
    MANUFACTURING = "manufacturing"
    FINANCIAL = "financial"
    TECHNOLOGY = "technology"
    RETAIL = "retail"
    UNKNOWN = "unknown"

@dataclass
class AuthorityContext:
    """
    Centralized authority context to eliminate scope issues.
    
    This class maintains all authority-related state throughout the
    analysis process and prevents the scope issues that were causing
    errors in the original enhanced_compliance.py file.
    """
    # Core detection results
    detected_authority: AuthorityType = AuthorityType.UNKNOWN
    detected_industry: IndustryType = IndustryType.UNKNOWN
    authority_confidence: float = 0.0
    
    # Content analysis
    german_content_detected: bool = False
    content_analysis_completed: bool = False
    
    # Authority analysis results
    authority_analysis: Optional[Any] = None
    authority_guidance: List[str] = field(default_factory=list)
    enforcement_patterns: Dict[str, Any] = field(default_factory=dict)
    
    # Compliance scoring
    audit_readiness_score: float = 0.0
    penalty_risk_level: str = "unknown"
    enforcement_likelihood: float = 0.0
    
    # Processing metadata
    detection_duration_ms: float = 0.0
    big4_engine_available: bool = False
    processing_errors: List[str] = field(default_factory=list)
    
    def has_valid_authority_data(self) -> bool:
        """Check if valid authority data was successfully detected"""
        return (
            self.detected_authority != AuthorityType.UNKNOWN and
            self.authority_confidence > 0.3 and
            self.content_analysis_completed
        )
    
    def has_german_content(self) -> bool:
        """Check if German content was detected in documents"""
        return self.german_content_detected
    
    def is_analysis_complete(self) -> bool:
        """Check if authority analysis is complete and valid"""
        return (
            self.content_analysis_completed and
            (not self.german_content_detected or self.authority_analysis is not None)
        )
    
    def get_risk_level(self) -> str:
        """Get simplified risk level for frontend display"""
        if self.penalty_risk_level in ["high", "critical"]:
            return "high"
        elif self.penalty_risk_level in ["medium", "moderate"]:
            return "medium"
        else:
            return "low"
    
    def to_api_response(self) -> Dict[str, Any]:
        """Convert to API response format with all relevant data"""
        return {
            "authority_detection": {
                "detected_authority": self.detected_authority.value,
                "detected_industry": self.detected_industry.value,
                "confidence": self.authority_confidence,
                "german_content_detected": self.german_content_detected
            },
            "compliance_assessment": {
                "audit_readiness_score": self.audit_readiness_score,
                "penalty_risk_level": self.penalty_risk_level,
                "enforcement_likelihood": self.enforcement_likelihood,
                "risk_level": self.get_risk_level()
            },
            "authority_guidance": {
                "recommendations": self.authority_guidance,
                "enforcement_patterns": self.enforcement_patterns
            },
            "processing_metadata": {
                "analysis_complete": self.is_analysis_complete(),
                "big4_engine_used": self.big4_engine_available,
                "detection_time_ms": self.detection_duration_ms,
                "errors": self.processing_errors
            }
        }

class AuthorityDetectionService:
    """
    Service for detecting German authorities and managing authority context.
    
    This service handles all authority-related functionality including:
    - German content detection
    - Authority identification based on content and location
    - Big 4 authority engine integration
    - Authority context management throughout analysis
    """
    
    def __init__(self):
        self.ui_context_layer = UIContextLayer()
        self.big4_endpoints: Optional[Big4AuthorityEndpoints] = None
        self._initialize_big4_engine()
        
        # Authority detection patterns
        self.authority_patterns = {
            AuthorityType.BFDI: [
                "bfdi", "bundesbeauftragte", "federal", "international transfer",
                "cross-border", "grenzüberschreitend", "international"
            ],
            AuthorityType.BAYLDA: [
                "baylda", "bayern", "bavaria", "munich", "münchen", 
                "bayerisches landesamt", "automotive", "bmw", "audi"
            ],
            AuthorityType.LFD_BW: [
                "lfd", "baden-württemberg", "stuttgart", "karlsruhe",
                "porsche", "mercedes", "bosch", "manufacturing"
            ],
            AuthorityType.LFD_NDS: [
                "niedersachsen", "hannover", "volkswagen", "wolfsburg",
                "continental", "automotive"
            ]
        }
        
        # Industry detection patterns
        self.industry_patterns = {
            IndustryType.AUTOMOTIVE: [
                "automotive", "car", "vehicle", "bmw", "mercedes", "audi",
                "porsche", "volkswagen", "fahrzeug", "automobil"
            ],
            IndustryType.HEALTHCARE: [
                "healthcare", "medical", "hospital", "patient", "gesundheit",
                "medizin", "klinik", "arzt", "health data", "pharma"
            ],
            IndustryType.MANUFACTURING: [
                "manufacturing", "production", "factory", "industrial",
                "herstellung", "produktion", "fertigung", "industrie"
            ],
            IndustryType.FINANCIAL: [
                "financial", "bank", "insurance", "payment", "fintech",
                "finanzen", "versicherung", "zahlung", "kredit"
            ],
            IndustryType.TECHNOLOGY: [
                "technology", "software", "app", "digital", "tech",
                "technologie", "software", "digital", "it", "saas"
            ],
            IndustryType.RETAIL: [
                "retail", "shop", "store", "e-commerce", "customer",
                "handel", "laden", "geschäft", "verkauf", "online"
            ]
        }
    
    def _initialize_big4_engine(self) -> None:
        """Initialize Big 4 Authority Engine with error handling"""
        try:
            self.big4_endpoints = create_big4_authority_endpoints()
            logger.info("Big 4 Authority Engine initialized successfully")
        except Exception as e:
            logger.warning(f"Big 4 Authority Engine initialization failed: {e}")
            self.big4_endpoints = None
    
    async def detect_authority_context(
        self,
        processed_files: List[Tuple[str, bytes, int]],
        company_location: Optional[str] = None,
        industry_hint: Optional[str] = None
    ) -> AuthorityContext:
        """
        Main entry point for authority detection and context creation.
        
        This method performs comprehensive authority detection and returns
        a complete AuthorityContext object with all relevant information.
        """
        import time
        start_time = time.time()
        
        context = AuthorityContext()
        context.big4_engine_available = self.big4_endpoints is not None
        
        try:
            # Step 1: Detect German content across all files
            context.german_content_detected = await self._detect_german_content(processed_files)
            
            # Step 2: Detect industry from content and hints
            context.detected_industry = await self._detect_industry(
                processed_files, industry_hint
            )
            
            # Step 3: Detect specific authority
            context.detected_authority, context.authority_confidence = await self._detect_specific_authority(
                processed_files, company_location, context.detected_industry
            )
            
            # Step 4: Perform Big 4 analysis if available and relevant
            if (context.german_content_detected and 
                context.detected_authority != AuthorityType.UNKNOWN and
                self.big4_endpoints):
                
                await self._perform_big4_analysis(context, processed_files)
            
            # Step 5: Generate authority guidance
            context.authority_guidance = await self._generate_authority_guidance(context)
            
            # Mark analysis as complete
            context.content_analysis_completed = True
            context.detection_duration_ms = (time.time() - start_time) * 1000
            
            logger.info(
                "Authority detection completed",
                authority=context.detected_authority.value,
                industry=context.detected_industry.value,
                confidence=context.authority_confidence,
                german_content=context.german_content_detected,
                duration_ms=context.detection_duration_ms
            )
            
        except Exception as e:
            error_msg = f"Authority detection failed: {str(e)}"
            context.processing_errors.append(error_msg)
            logger.error(error_msg, exc_info=True)
        
        return context
    
    async def _detect_german_content(self, processed_files: List[Tuple[str, bytes, int]]) -> bool:
        """Detect if any files contain German content"""
        try:
            for filename, content, _ in processed_files:
                if self.ui_context_layer.detect_german_content(content):
                    return True
            return False
        except Exception as e:
            logger.warning(f"German content detection failed: {e}")
            return False
    
    async def _detect_industry(
        self,
        processed_files: List[Tuple[str, bytes, int]],
        industry_hint: Optional[str]
    ) -> IndustryType:
        """Detect industry type from content and hints"""
        
        # Check explicit hint first
        if industry_hint:
            try:
                return IndustryType(industry_hint.lower())
            except ValueError:
                pass
        
        # Analyze content for industry indicators
        industry_scores = {industry: 0 for industry in IndustryType}
        
        for filename, content, _ in processed_files:
            try:
                # Convert content to text for analysis
                text = content.decode('utf-8', errors='ignore').lower()
                
                # Score based on keyword matches
                for industry, keywords in self.industry_patterns.items():
                    for keyword in keywords:
                        if keyword in text:
                            industry_scores[industry] += 1
                
                # Filename-based scoring
                filename_lower = filename.lower()
                for industry, keywords in self.industry_patterns.items():
                    for keyword in keywords:
                        if keyword in filename_lower:
                            industry_scores[industry] += 2  # Filename matches weighted higher
                            
            except Exception as e:
                logger.warning(f"Industry detection failed for {filename}: {e}")
        
        # Return industry with highest score
        if max(industry_scores.values()) > 0:
            return max(industry_scores, key=industry_scores.get)
        
        return IndustryType.UNKNOWN
    
    async def _detect_specific_authority(
        self,
        processed_files: List[Tuple[str, bytes, int]],
        company_location: Optional[str],
        detected_industry: IndustryType
    ) -> Tuple[AuthorityType, float]:
        """Detect specific German authority based on content and location"""
        
        authority_scores = {authority: 0.0 for authority in AuthorityType}
        
        # Location-based scoring
        if company_location:
            location_lower = company_location.lower()
            if any(term in location_lower for term in ["bayern", "bavaria", "munich", "münchen"]):
                authority_scores[AuthorityType.BAYLDA] += 0.3
            elif any(term in location_lower for term in ["baden-württemberg", "stuttgart", "karlsruhe"]):
                authority_scores[AuthorityType.LFD_BW] += 0.3
            elif any(term in location_lower for term in ["niedersachsen", "hannover", "wolfsburg"]):
                authority_scores[AuthorityType.LFD_NDS] += 0.3
        
        # Industry-based scoring
        if detected_industry == IndustryType.AUTOMOTIVE:
            authority_scores[AuthorityType.BAYLDA] += 0.2
            authority_scores[AuthorityType.LFD_BW] += 0.2
            authority_scores[AuthorityType.LFD_NDS] += 0.1
        
        # Content-based scoring
        for filename, content, _ in processed_files:
            try:
                text = content.decode('utf-8', errors='ignore').lower()
                
                for authority, keywords in self.authority_patterns.items():
                    for keyword in keywords:
                        if keyword in text:
                            authority_scores[authority] += 0.1
                            
            except Exception as e:
                logger.warning(f"Authority content analysis failed for {filename}: {e}")
        
        # International transfer detection (BfDI specialty)
        for filename, content, _ in processed_files:
            try:
                text = content.decode('utf-8', errors='ignore').lower()
                international_indicators = [
                    "international transfer", "adequacy decision", "standard contractual clauses",
                    "third country", "drittland", "angemessenheitsbeschluss"
                ]
                
                for indicator in international_indicators:
                    if indicator in text:
                        authority_scores[AuthorityType.BFDI] += 0.2
                        
            except Exception:
                pass
        
        # Return authority with highest score and confidence
        if max(authority_scores.values()) > 0:
            best_authority = max(authority_scores, key=authority_scores.get)
            confidence = min(1.0, authority_scores[best_authority])
            return best_authority, confidence
        
        return AuthorityType.UNKNOWN, 0.0
    
    async def _perform_big4_analysis(
        self,
        context: AuthorityContext,
        processed_files: List[Tuple[str, bytes, int]]
    ) -> None:
        """Perform Big 4 authority analysis if engine is available"""
        
        if not self.big4_endpoints:
            return
        
        try:
            # Get authority-specific analysis
            authority_analysis = await self.big4_endpoints.get_authority_analysis(
                authority=context.detected_authority.value,
                industry=context.detected_industry.value,
                document_count=len(processed_files)
            )
            
            context.authority_analysis = authority_analysis
            
            # Extract specific metrics if available
            if hasattr(authority_analysis, 'audit_readiness_score'):
                context.audit_readiness_score = authority_analysis.audit_readiness_score
            
            if hasattr(authority_analysis, 'penalty_risk_level'):
                context.penalty_risk_level = authority_analysis.penalty_risk_level
            
            if hasattr(authority_analysis, 'enforcement_likelihood'):
                context.enforcement_likelihood = authority_analysis.enforcement_likelihood
            
            if hasattr(authority_analysis, 'enforcement_patterns'):
                context.enforcement_patterns = authority_analysis.enforcement_patterns
                
        except Exception as e:
            error_msg = f"Big 4 analysis failed: {str(e)}"
            context.processing_errors.append(error_msg)
            logger.warning(error_msg)
    
    async def _generate_authority_guidance(self, context: AuthorityContext) -> List[str]:
        """Generate authority-specific guidance recommendations"""
        
        guidance = []
        
        # Authority-specific guidance
        if context.detected_authority == AuthorityType.BFDI:
            guidance.extend([
                "Focus on international data transfer compliance",
                "Ensure adequacy decisions or SCCs are in place",
                "Document cross-border processing activities"
            ])
        elif context.detected_authority == AuthorityType.BAYLDA:
            guidance.extend([
                "Prepare for technical audit approach",
                "Document privacy by design implementation", 
                "Focus on automotive industry standards"
            ])
        elif context.detected_authority == AuthorityType.LFD_BW:
            guidance.extend([
                "Emphasize risk-based compliance approach",
                "Prepare comprehensive documentation",
                "Focus on manufacturing data protection"
            ])
        
        # Industry-specific guidance
        if context.detected_industry == IndustryType.AUTOMOTIVE:
            guidance.extend([
                "Address connected vehicle data protection",
                "Implement supply chain data governance",
                "Ensure customer consent for telematics"
            ])
        elif context.detected_industry == IndustryType.HEALTHCARE:
            guidance.extend([
                "Focus on patient data protection measures",
                "Implement medical data anonymization",
                "Ensure research data compliance"
            ])
        
        # Risk-based guidance
        if context.penalty_risk_level in ["high", "critical"]:
            guidance.append("Consider immediate compliance audit")
            guidance.append("Implement enhanced monitoring measures")
        
        return guidance[:10]  # Limit to top 10 recommendations