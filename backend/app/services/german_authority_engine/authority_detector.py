# app/services/german_authority_engine/authority_detector.py
"""
Authority Detector

Automatically detects which German data protection authorities are most
relevant based on document content, industry context, and geographic indicators.
"""

from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass
import structlog

from app.models.database import Document
from .authority_profiles import (
    GermanAuthority, AuthorityProfile, 
    get_all_authorities, get_authorities_by_industry, get_authorities_by_region
)

logger = structlog.get_logger()

@dataclass
class AuthorityDetectionResult:
    """Result of authority detection analysis"""
    detected_authorities: List[GermanAuthority]
    detection_confidence: Dict[str, float]
    detection_reasons: Dict[str, List[str]]
    geographic_indicators: List[str]
    industry_indicators: List[str]
    content_indicators: Dict[str, List[str]]

class AuthorityDetector:
    """
    Intelligent German authority detection based on content analysis
    
    Analyzes document content, industry context, and geographic indicators
    to determine which German DPAs are most relevant for compliance analysis.
    """
    
    def __init__(self):
        # Geographic indicators for German states
        self.geographic_indicators = {
            GermanAuthority.BAYERN: {
                "cities": ["münchen", "munich", "nuremberg", "nürnberg", "augsburg", "ingolstadt", "ansbach"],
                "regions": ["bavaria", "bayern", "bavarian", "bayerisch"],
                "postal_codes": ["8", "9"],  # Bavarian postal code prefixes
                "keywords": ["bayern", "bavaria", "bayerisch", "bavarian"]
            },
            GermanAuthority.BADEN_WURTTEMBERG: {
                "cities": ["stuttgart", "karlsruhe", "mannheim", "freiburg", "ulm", "heilbronn"],
                "regions": ["baden-württemberg", "baden", "württemberg", "bw"],
                "postal_codes": ["6", "7"],
                "keywords": ["baden-württemberg", "stuttgart", "schwäbisch"]
            },
            GermanAuthority.NORDRHEIN_WESTFALEN: {
                "cities": ["köln", "cologne", "düsseldorf", "dortmund", "essen", "duisburg", "bochum"],
                "regions": ["nordrhein-westfalen", "nrw", "rheinland", "westfalen", "ruhr"],
                "postal_codes": ["4", "5"],
                "keywords": ["nrw", "nordrhein", "westfalen", "rheinland"]
            },
            GermanAuthority.HESSEN: {
                "cities": ["frankfurt", "wiesbaden", "kassel", "darmstadt", "offenbach"],
                "regions": ["hessen", "hesse", "hessisch"],
                "postal_codes": ["6"],
                "keywords": ["hessen", "hesse", "frankfurt", "rhein-main"]
            },
            GermanAuthority.BERLIN: {
                "cities": ["berlin"],
                "regions": ["berlin", "hauptstadt"],
                "postal_codes": ["1"],
                "keywords": ["berlin", "hauptstadt", "bundestag"]
            }
        }
        
        # Industry-specific authority mappings
        self.industry_authority_mapping = {
            "automotive": [GermanAuthority.BAYERN, GermanAuthority.BADEN_WURTTEMBERG],
            "manufacturing": [GermanAuthority.BAYERN, GermanAuthority.BADEN_WURTTEMBERG, GermanAuthority.NORDRHEIN_WESTFALEN],
            "financial": [GermanAuthority.HESSEN, GermanAuthority.BERLIN, GermanAuthority.BFDI],
            "technology": [GermanAuthority.BERLIN, GermanAuthority.BAYERN, GermanAuthority.BADEN_WURTTEMBERG],
            "healthcare": [GermanAuthority.BFDI, GermanAuthority.NORDRHEIN_WESTFALEN],
            "logistics": [GermanAuthority.NORDRHEIN_WESTFALEN, GermanAuthority.RHEINLAND_PFALZ],
            "international": [GermanAuthority.BFDI]
        }
        
        # Content-based detection patterns
        self.content_patterns = {
            "federal_indicators": [
                "bundesregierung", "federal government", "international transfer",
                "grenzüberschreitend", "cross-border", "adequacy decision",
                "standard contractual clauses", "sccs", "bcr"
            ],
            "automotive_indicators": [
                "fahrzeug", "vehicle", "automotive", "telematics", "connected car",
                "bmw", "audi", "mercedes", "porsche", "bosch", "continental"
            ],
            "manufacturing_indicators": [
                "produktion", "manufacturing", "fertigung", "industrie 4.0",
                "maschinenbau", "engineering", "fabrik", "factory"
            ],
            "financial_indicators": [
                "bank", "financial", "fintech", "payment", "kredit", "versicherung",
                "bafin", "banking", "investment", "trading"
            ]
        }
    
    async def detect_relevant_authorities(
        self,
        documents: List[Document],
        industry: Optional[str] = None,
        location: Optional[str] = None
    ) -> List[GermanAuthority]:
        """
        Detect most relevant German authorities for document set
        
        Args:
            documents: Documents to analyze for authority relevance
            industry: Optional industry context
            location: Optional geographic location
            
        Returns:
            List of relevant German authorities in priority order
        """
        logger.info(
            "Starting authority detection",
            documents=len(documents),
            industry=industry,
            location=location
        )
        
        # Extract document content for analysis
        document_content = await self._extract_content_for_analysis(documents)
        
        # Perform detection analysis
        detection_result = await self._analyze_authority_relevance(
            document_content, industry, location
        )
        
        # Rank authorities by relevance
        ranked_authorities = self._rank_authorities_by_relevance(detection_result)
        
        logger.info(
            "Authority detection completed",
            detected_authorities=len(ranked_authorities),
            top_authority=ranked_authorities[0].value if ranked_authorities else None
        )
        
        return ranked_authorities
    
    async def get_detection_analysis(
        self,
        documents: List[Document],
        industry: Optional[str] = None,
        location: Optional[str] = None
    ) -> AuthorityDetectionResult:
        """
        Get detailed authority detection analysis with reasoning
        
        Returns comprehensive analysis showing why specific authorities
        were detected and confidence levels.
        """
        document_content = await self._extract_content_for_analysis(documents)
        
        return await self._analyze_authority_relevance(document_content, industry, location)
    
    async def _extract_content_for_analysis(self, documents: List[Document]) -> str:
        """Extract and combine document content for authority detection"""
        
        all_content = []
        
        for doc in documents:
            # Get content from chunks if available
            if hasattr(doc, 'chunks') and doc.chunks:
                doc_content = " ".join([
                    chunk.content for chunk in doc.chunks
                    if hasattr(chunk, 'content') and chunk.content
                ])
            else:
                # Fallback to document-level content
                doc_content = getattr(doc, 'content', '') or doc.filename
            
            all_content.append(doc_content)
        
        return " ".join(all_content).lower()
    
    async def _analyze_authority_relevance(
        self,
        content: str,
        industry: Optional[str],
        location: Optional[str]
    ) -> AuthorityDetectionResult:
        """Analyze content to determine authority relevance"""
        
        authority_scores = {}
        detection_reasons = {}
        geographic_indicators = []
        industry_indicators = []
        content_indicators = {}
        
        all_authorities = get_all_authorities()
        
        # Initialize scoring for all authorities
        for authority in all_authorities.keys():
            authority_scores[authority.value] = 0.0
            detection_reasons[authority.value] = []
            content_indicators[authority.value] = []
        
        # 1. Geographic detection
        geographic_scores = self._detect_geographic_relevance(content, location)
        for authority_id, score in geographic_scores.items():
            authority_scores[authority_id] += score * 0.4  # 40% weight
            if score > 0:
                detection_reasons[authority_id].append(f"Geographic relevance: {score:.2f}")
        
        # 2. Industry-based detection
        industry_scores = self._detect_industry_relevance(content, industry)
        for authority_id, score in industry_scores.items():
            authority_scores[authority_id] += score * 0.3  # 30% weight
            if score > 0:
                detection_reasons[authority_id].append(f"Industry relevance: {score:.2f}")
        
        # 3. Content-based detection
        content_scores = self._detect_content_relevance(content)
        for authority_id, (score, indicators) in content_scores.items():
            authority_scores[authority_id] += score * 0.3  # 30% weight
            content_indicators[authority_id] = indicators
            if score > 0:
                detection_reasons[authority_id].append(f"Content relevance: {score:.2f}")
        
        # Extract top indicators for summary
        geographic_indicators = self._extract_geographic_indicators(content)
        industry_indicators = self._extract_industry_indicators(content, industry)
        
        # Convert scores to confidence levels (0-1)
        confidence_levels = {
            authority_id: min(1.0, score) 
            for authority_id, score in authority_scores.items()
        }
        
        # Get detected authorities (above threshold)
        detected_authorities = [
            GermanAuthority(authority_id)
            for authority_id, confidence in confidence_levels.items()
            if confidence >= 0.2  # 20% minimum confidence threshold
        ]
        
        return AuthorityDetectionResult(
            detected_authorities=detected_authorities,
            detection_confidence=confidence_levels,
            detection_reasons=detection_reasons,
            geographic_indicators=geographic_indicators,
            industry_indicators=industry_indicators,
            content_indicators=content_indicators
        )
    
    def _detect_geographic_relevance(self, content: str, location: Optional[str]) -> Dict[str, float]:
        """Detect geographic relevance for authorities"""
        
        geographic_scores = {}
        
        for authority, indicators in self.geographic_indicators.items():
            score = 0.0
            
            # Check explicit location match
            if location:
                location_lower = location.lower()
                if any(city in location_lower for city in indicators["cities"]):
                    score += 0.8
                elif any(region in location_lower for region in indicators["regions"]):
                    score += 0.6
            
            # Check content for geographic indicators
            city_matches = sum(1 for city in indicators["cities"] if city in content)
            region_matches = sum(1 for region in indicators["regions"] if region in content)
            keyword_matches = sum(1 for keyword in indicators["keywords"] if keyword in content)
            
            # Calculate content-based geographic score
            content_score = (
                city_matches * 0.3 +
                region_matches * 0.2 +
                keyword_matches * 0.1
            )
            
            score += min(0.5, content_score)  # Cap content contribution at 0.5
            
            geographic_scores[authority.value] = score
        
        return geographic_scores
    
    def _detect_industry_relevance(self, content: str, industry: Optional[str]) -> Dict[str, float]:
        """Detect industry-based authority relevance"""
        
        industry_scores = {}
        
        # Initialize all authorities with base score
        all_authorities = get_all_authorities()
        for authority in all_authorities.keys():
            industry_scores[authority.value] = 0.0
        
        # Explicit industry mapping
        if industry and industry.lower() in self.industry_authority_mapping:
            relevant_authorities = self.industry_authority_mapping[industry.lower()]
            for authority in relevant_authorities:
                industry_scores[authority.value] += 0.7
        
        # Content-based industry detection
        for industry_type, authorities in self.industry_authority_mapping.items():
            if industry_type in self.content_patterns:
                indicators = self.content_patterns[f"{industry_type}_indicators"]
                matches = sum(1 for indicator in indicators if indicator in content)
                
                if matches > 0:
                    match_score = min(0.5, matches * 0.1)
                    for authority in authorities:
                        industry_scores[authority.value] += match_score
        
        return industry_scores
    
    def _detect_content_relevance(self, content: str) -> Dict[str, tuple]:
        """Detect content-based authority relevance"""
        
        content_scores = {}
        all_authorities = get_all_authorities()
        
        for authority in all_authorities.keys():
            score = 0.0
            found_indicators = []
            
            # Check for federal indicators
            if authority == GermanAuthority.BFDI:
                federal_matches = sum(
                    1 for indicator in self.content_patterns["federal_indicators"]
                    if indicator in content
                )
                if federal_matches > 0:
                    score += min(0.6, federal_matches * 0.15)
                    found_indicators.extend([
                        indicator for indicator in self.content_patterns["federal_indicators"]
                        if indicator in content
                    ])
            
            # Check for automotive indicators (Bayern, Baden-Württemberg)
            if authority in [GermanAuthority.BAYERN, GermanAuthority.BADEN_WURTTEMBERG]:
                automotive_matches = sum(
                    1 for indicator in self.content_patterns["automotive_indicators"]
                    if indicator in content
                )
                if automotive_matches > 0:
                    score += min(0.5, automotive_matches * 0.1)
                    found_indicators.extend([
                        indicator for indicator in self.content_patterns["automotive_indicators"]
                        if indicator in content
                    ])
            
            content_scores[authority.value] = (score, found_indicators[:5])  # Limit indicators
        
        return content_scores
    
    def _extract_geographic_indicators(self, content: str) -> List[str]:
        """Extract geographic indicators found in content"""
        
        indicators = []
        for authority, data in self.geographic_indicators.items():
            for city in data["cities"]:
                if city in content:
                    indicators.append(f"City: {city}")
            for region in data["regions"]:
                if region in content:
                    indicators.append(f"Region: {region}")
        
        return list(set(indicators))[:10]  # Limit and deduplicate
    
    def _extract_industry_indicators(self, content: str, industry: Optional[str]) -> List[str]:
        """Extract industry indicators from content"""
        
        indicators = []
        
        if industry:
            indicators.append(f"Explicit industry: {industry}")
        
        for industry_type, patterns in self.content_patterns.items():
            if industry_type.endswith("_indicators"):
                industry_name = industry_type.replace("_indicators", "")
                matches = [pattern for pattern in patterns if pattern in content]
                if matches:
                    indicators.append(f"{industry_name.title()}: {len(matches)} indicators")
        
        return indicators[:8]  # Limit results
    
    def _rank_authorities_by_relevance(self, detection_result: AuthorityDetectionResult) -> List[GermanAuthority]:
        """Rank authorities by detection confidence"""
        
        # Sort authorities by confidence level
        sorted_authorities = sorted(
            detection_result.detection_confidence.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Filter authorities with meaningful confidence (>= 0.2)
        relevant_authorities = [
            GermanAuthority(authority_id)
            for authority_id, confidence in sorted_authorities
            if confidence >= 0.2
        ]
        
        # Ensure federal authority is included if no specific authorities detected
        if not relevant_authorities:
            relevant_authorities = [GermanAuthority.BFDI]
        
        # Limit to top 5 most relevant authorities
        return relevant_authorities[:5]
    
    async def suggest_authority_for_industry_location(
        self,
        industry: str,
        location: str
    ) -> List[GermanAuthority]:
        """
        Suggest authorities based on industry and location combination
        
        Useful for new compliance projects where no documents exist yet.
        """
        logger.info(
            "Suggesting authorities for industry and location",
            industry=industry,
            location=location
        )
        
        suggestions = []
        
        # Industry-based suggestions
        if industry.lower() in self.industry_authority_mapping:
            suggestions.extend(self.industry_authority_mapping[industry.lower()])
        
        # Location-based suggestions
        location_lower = location.lower()
        for authority, indicators in self.geographic_indicators.items():
            if (any(city in location_lower for city in indicators["cities"]) or
                any(region in location_lower for region in indicators["regions"])):
                if authority not in suggestions:
                    suggestions.append(authority)
        
        # Always include federal authority for comprehensive coverage
        if GermanAuthority.BFDI not in suggestions:
            suggestions.append(GermanAuthority.BFDI)
        
        return suggestions[:3]  # Return top 3 suggestions