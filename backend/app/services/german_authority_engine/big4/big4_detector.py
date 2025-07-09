# backend/app/services/german_authority_engine/big4/big4_detector.py
"""
Big 4 Authority Intelligent Detection

Smart detection of relevant German authorities based on:
- Document content analysis
- Company location and industry
- Business context and size
- Compliance pattern recognition

Uses machine learning-like scoring to recommend the most relevant authorities.
"""

import re
from dataclasses import dataclass
from typing import List, Dict, Optional, Set, Tuple
from collections import defaultdict
import structlog

from app.models.database import Document
from .big4_profiles import (
    Big4Authority, Big4AuthorityProfile, get_big4_authority_profile,
    get_authorities_by_industry, get_authority_by_state,
    INDUSTRY_AUTHORITY_MAPPING, STATE_AUTHORITY_MAPPING
)

logger = structlog.get_logger()

@dataclass
class DetectionScore:
    """Authority detection scoring breakdown"""
    authority: Big4Authority
    total_score: float
    confidence: float
    reasoning: List[str]
    
    # Score components
    geographic_score: float
    industry_score: float
    content_score: float
    business_context_score: float

@dataclass
class Big4DetectionResult:
    """Complete authority detection analysis"""
    primary_authority: Big4Authority
    all_scores: List[DetectionScore]
    detection_confidence: float
    
    # Detection insights
    geographic_indicators: List[str]
    industry_indicators: List[str] 
    content_indicators: Dict[str, List[str]]
    
    # Recommendations
    recommended_analysis_order: List[Big4Authority]
    multi_authority_needed: bool
    fallback_suggestion: Optional[str]

class Big4AuthorityDetector:
    """
    Intelligent detector for Big 4 German authorities
    
    Uses content analysis, geographic indicators, and business context
    to recommend the most relevant authorities for compliance analysis.
    """
    
    def __init__(self):
        self.geographic_patterns = self._load_geographic_patterns()
        self.industry_patterns = self._load_industry_patterns()
        self.content_patterns = self._load_content_patterns()
        self.business_context_patterns = self._load_business_context_patterns()
        
        logger.info("Big 4 Authority Detector initialized")
    
    async def detect_relevant_authorities(
        self,
        documents: List[Document],
        suggested_industry: Optional[str] = None,
        suggested_state: Optional[str] = None,
        company_size: Optional[str] = None
    ) -> Big4DetectionResult:
        """
        Detect most relevant Big 4 authorities for documents
        
        Args:
            documents: Documents to analyze
            suggested_industry: User-provided industry hint
            suggested_state: User-provided state/location hint
            company_size: small, medium, large
            
        Returns:
            Complete detection analysis with scored recommendations
        """
        
        logger.info(
            "Starting Big 4 authority detection",
            documents=len(documents),
            suggested_industry=suggested_industry,
            suggested_state=suggested_state
        )
        
        # Analyze all documents for detection signals
        combined_content = self._combine_document_content(documents)
        
        # Extract detection indicators
        geographic_indicators = self._extract_geographic_indicators(combined_content, suggested_state)
        industry_indicators = self._extract_industry_indicators(combined_content, suggested_industry)
        content_indicators = self._extract_content_indicators(combined_content)
        
        # Score each Big 4 authority
        authority_scores = []
        for authority in Big4Authority:
            score = await self._score_authority_relevance(
                authority=authority,
                geographic_indicators=geographic_indicators,
                industry_indicators=industry_indicators,
                content_indicators=content_indicators,
                company_size=company_size,
                document_count=len(documents)
            )
            authority_scores.append(score)
        
        # Sort by total score (highest first)
        authority_scores.sort(key=lambda x: x.total_score, reverse=True)
        
        # Determine primary authority and confidence
        primary_authority = authority_scores[0].authority
        detection_confidence = self._calculate_detection_confidence(authority_scores)
        
        # Create recommendations
        recommended_order = [score.authority for score in authority_scores[:3]]
        multi_authority_needed = self._should_use_multi_authority_analysis(authority_scores)
        fallback_suggestion = self._generate_fallback_suggestion(authority_scores, detection_confidence)
        
        result = Big4DetectionResult(
            primary_authority=primary_authority,
            all_scores=authority_scores,
            detection_confidence=detection_confidence,
            geographic_indicators=geographic_indicators,
            industry_indicators=industry_indicators,
            content_indicators=content_indicators,
            recommended_analysis_order=recommended_order,
            multi_authority_needed=multi_authority_needed,
            fallback_suggestion=fallback_suggestion
        )
        
        logger.info(
            "Big 4 authority detection completed",
            primary_authority=primary_authority.value,
            confidence=detection_confidence,
            multi_authority_needed=multi_authority_needed
        )
        
        return result
    
    async def suggest_authorities_for_business(
        self,
        location: str,
        industry: str,
        company_size: str,
        business_activities: Optional[List[str]] = None
    ) -> Big4DetectionResult:
        """
        Suggest authorities based on business profile (no documents)
        
        Useful for onboarding and initial authority selection.
        """
        
        # Create synthetic indicators based on business profile
        geographic_indicators = [location.lower()]
        industry_indicators = [industry.lower()]
        
        # Enhanced industry detection from business activities
        if business_activities:
            for activity in business_activities:
                industry_indicators.extend(self._extract_industry_from_activity(activity))
        
        content_indicators = {"business_profile": [industry, location, company_size]}
        
        # Score authorities based on business profile
        authority_scores = []
        for authority in Big4Authority:
            score = await self._score_authority_relevance(
                authority=authority,
                geographic_indicators=geographic_indicators,
                industry_indicators=industry_indicators,
                content_indicators=content_indicators,
                company_size=company_size,
                document_count=0  # No documents in business profile mode
            )
            authority_scores.append(score)
        
        authority_scores.sort(key=lambda x: x.total_score, reverse=True)
        
        # Business profile confidence is generally lower than document analysis
        detection_confidence = min(0.8, self._calculate_detection_confidence(authority_scores))
        
        return Big4DetectionResult(
            primary_authority=authority_scores[0].authority,
            all_scores=authority_scores,
            detection_confidence=detection_confidence,
            geographic_indicators=geographic_indicators,
            industry_indicators=industry_indicators,
            content_indicators=content_indicators,
            recommended_analysis_order=[score.authority for score in authority_scores[:3]],
            multi_authority_needed=len([s for s in authority_scores if s.total_score > 0.6]) > 1,
            fallback_suggestion="Consider document analysis for more precise authority detection"
        )
    
    async def _score_authority_relevance(
        self,
        authority: Big4Authority,
        geographic_indicators: List[str],
        industry_indicators: List[str],
        content_indicators: Dict[str, List[str]],
        company_size: Optional[str],
        document_count: int
    ) -> DetectionScore:
        """Score how relevant an authority is for the given context"""
        
        authority_profile = get_big4_authority_profile(authority)
        reasoning = []
        
        # Geographic scoring (0-1)
        geographic_score = self._score_geographic_relevance(
            authority, authority_profile, geographic_indicators, reasoning
        )
        
        # Industry scoring (0-1)
        industry_score = self._score_industry_relevance(
            authority, authority_profile, industry_indicators, reasoning
        )
        
        # Content scoring (0-1)
        content_score = self._score_content_relevance(
            authority, authority_profile, content_indicators, reasoning
        )
        
        # Business context scoring (0-1)
        business_context_score = self._score_business_context_relevance(
            authority, authority_profile, company_size, document_count, reasoning
        )
        
        # Weighted total score
        weights = {
            "geographic": 0.30,
            "industry": 0.35,
            "content": 0.25,
            "business_context": 0.10
        }
        
        total_score = (
            geographic_score * weights["geographic"] +
            industry_score * weights["industry"] + 
            content_score * weights["content"] +
            business_context_score * weights["business_context"]
        )
        
        # Calculate confidence based on signal strength
        confidence = self._calculate_score_confidence(
            geographic_score, industry_score, content_score, business_context_score
        )
        
        return DetectionScore(
            authority=authority,
            total_score=total_score,
            confidence=confidence,
            reasoning=reasoning,
            geographic_score=geographic_score,
            industry_score=industry_score,
            content_score=content_score,
            business_context_score=business_context_score
        )
    
    def _score_geographic_relevance(
        self,
        authority: Big4Authority,
        profile: Big4AuthorityProfile,
        indicators: List[str],
        reasoning: List[str]
    ) -> float:
        """Score geographic relevance (0-1)"""
        
        score = 0.0
        
        # Direct state mapping
        for indicator in indicators:
            if indicator in STATE_AUTHORITY_MAPPING:
                mapped_authority = STATE_AUTHORITY_MAPPING[indicator]
                if mapped_authority == authority:
                    score = 1.0
                    reasoning.append(f"Direct geographic match: {indicator} → {authority.value}")
                    break
        
        # Jurisdiction-based scoring
        if score == 0.0:
            jurisdiction_keywords = {
                Big4Authority.BFDI: ["federal", "deutschland", "germany", "bund"],
                Big4Authority.BAYLDA: ["bayern", "bavaria", "munich", "münchen", "nuremberg"],
                Big4Authority.LFD_BW: ["baden", "württemberg", "stuttgart", "karlsruhe"],
                Big4Authority.LDI_NRW: ["nordrhein", "westfalen", "düsseldorf", "köln", "ruhr"]
            }
            
            authority_keywords = jurisdiction_keywords.get(authority, [])
            matches = sum(1 for indicator in indicators for keyword in authority_keywords
                         if keyword in indicator.lower())
            
            if matches > 0:
                score = min(1.0, matches * 0.3)
                reasoning.append(f"Geographic keywords matched: {matches}")
        
        # Default federal authority fallback
        if score == 0.0 and authority == Big4Authority.BFDI:
            score = 0.1
            reasoning.append("Federal authority default fallback")
        
        return score
    
    def _score_industry_relevance(
        self,
        authority: Big4Authority,
        profile: Big4AuthorityProfile,
        indicators: List[str],
        reasoning: List[str]
    ) -> float:
        """Score industry relevance (0-1)"""
        
        score = 0.0
        
        # Direct industry specialization match
        for indicator in indicators:
            if indicator in profile.industry_specializations:
                score = max(score, 1.0)
                reasoning.append(f"Direct industry specialization: {indicator}")
            elif indicator in profile.enforcement_profile.industry_focus_areas:
                score = max(score, 0.8)
                reasoning.append(f"Industry focus area: {indicator}")
        
        # Industry mapping scoring
        if score < 1.0:
            for indicator in indicators:
                if indicator in INDUSTRY_AUTHORITY_MAPPING:
                    relevant_authorities = INDUSTRY_AUTHORITY_MAPPING[indicator]
                    if authority in relevant_authorities:
                        position_score = 1.0 - (relevant_authorities.index(authority) * 0.2)
                        score = max(score, position_score)
                        reasoning.append(f"Industry mapping: {indicator} (position {relevant_authorities.index(authority) + 1})")
        
        return score
    
    def _score_content_relevance(
        self,
        authority: Big4Authority,
        profile: Big4AuthorityProfile,
        indicators: Dict[str, List[str]],
        reasoning: List[str]
    ) -> float:
        """Score content relevance based on document analysis (0-1)"""
        
        score = 0.0
        
        # Priority GDPR articles mentioned
        if "gdpr_articles" in indicators:
            article_matches = sum(
                1 for article in indicators["gdpr_articles"]
                if article in profile.priority_articles
            )
            if article_matches > 0:
                article_score = min(1.0, article_matches * 0.2)
                score = max(score, article_score)
                reasoning.append(f"Priority GDPR articles matched: {article_matches}")
        
        # Unique requirements mentioned
        if "compliance_terms" in indicators:
            requirement_matches = 0
            for term in indicators["compliance_terms"]:
                for requirement in profile.unique_requirements:
                    if any(word in requirement.lower() for word in term.lower().split()):
                        requirement_matches += 1
            
            if requirement_matches > 0:
                requirement_score = min(1.0, requirement_matches * 0.3)
                score = max(score, requirement_score)
                reasoning.append(f"Unique requirements matched: {requirement_matches}")
        
        return score
    
    def _score_business_context_relevance(
        self,
        authority: Big4Authority,
        profile: Big4AuthorityProfile,
        company_size: Optional[str],
        document_count: int,
        reasoning: List[str]
    ) -> float:
        """Score business context relevance (0-1)"""
        
        score = 0.0
        
        # SME focus scoring
        if company_size in ["small", "medium"] and profile.sme_focus:
            score = max(score, 0.8)
            reasoning.append(f"SME-focused authority for {company_size} company")
        elif company_size == "large" and not profile.sme_focus:
            score = max(score, 0.6)
            reasoning.append("Large enterprise authority match")
        
        # Document count context
        if document_count > 5:  # Complex compliance portfolio
            if authority == Big4Authority.BFDI:  # Federal authority for complex cases
                score = max(score, 0.4)
                reasoning.append("Complex portfolio suggests federal oversight")
        
        # Market share consideration
        market_share_score = profile.market_share * 0.5  # Up to 0.15 for BfDI
        score = max(score, market_share_score)
        
        return score
    
    def _calculate_score_confidence(
        self,
        geographic_score: float,
        industry_score: float,
        content_score: float,
        business_context_score: float
    ) -> float:
        """Calculate confidence in the score (0-1)"""
        
        # Confidence is higher when multiple signals align
        non_zero_scores = sum(1 for score in [geographic_score, industry_score, content_score, business_context_score] if score > 0.1)
        
        # Base confidence from signal count
        base_confidence = min(1.0, non_zero_scores * 0.25)
        
        # Boost confidence for strong signals
        strong_signals = sum(1 for score in [geographic_score, industry_score, content_score, business_context_score] if score > 0.7)
        confidence_boost = strong_signals * 0.2
        
        return min(1.0, base_confidence + confidence_boost)
    
    def _calculate_detection_confidence(self, authority_scores: List[DetectionScore]) -> float:
        """Calculate overall detection confidence"""
        
        if not authority_scores:
            return 0.0
        
        # Confidence based on score separation
        top_score = authority_scores[0].total_score
        second_score = authority_scores[1].total_score if len(authority_scores) > 1 else 0.0
        
        score_separation = top_score - second_score
        separation_confidence = min(1.0, score_separation * 2)  # More separation = higher confidence
        
        # Confidence based on absolute score
        absolute_confidence = min(1.0, top_score)
        
        # Combined confidence
        return min(1.0, (separation_confidence + absolute_confidence) / 2)
    
    def _should_use_multi_authority_analysis(self, authority_scores: List[DetectionScore]) -> bool:
        """Determine if multi-authority analysis is recommended"""
        
        if len(authority_scores) < 2:
            return False
        
        # Multi-authority if top 2 scores are close
        top_score = authority_scores[0].total_score
        second_score = authority_scores[1].total_score
        
        return (second_score / max(top_score, 0.1)) > 0.7  # Within 30% of top score
    
    def _generate_fallback_suggestion(
        self,
        authority_scores: List[DetectionScore],
        confidence: float
    ) -> Optional[str]:
        """Generate fallback suggestion if confidence is low"""
        
        if confidence > 0.6:
            return None
        
        suggestions = []
        
        if confidence < 0.3:
            suggestions.append("Consider providing company location and industry for better detection")
        
        if all(score.total_score < 0.4 for score in authority_scores):
            suggestions.append("Default to BfDI (federal authority) for broad compliance coverage")
        
        return "; ".join(suggestions) if suggestions else None
    
    # Pattern loading methods
    def _load_geographic_patterns(self) -> Dict[str, List[str]]:
        """Load geographic detection patterns"""
        return {
            "bayern": ["bayern", "bavaria", "munich", "münchen", "nuremberg", "nürnberg", "augsburg"],
            "baden_wurttemberg": ["baden", "württemberg", "stuttgart", "karlsruhe", "mannheim", "heidelberg"],
            "nordrhein_westfalen": ["nordrhein", "westfalen", "nrw", "düsseldorf", "köln", "dortmund", "essen", "ruhr"],
            "federal": ["deutschland", "germany", "federal", "bund", "bundesweit"]
        }
    
    def _load_industry_patterns(self) -> Dict[str, List[str]]:
        """Load industry detection patterns"""
        return {
            "automotive": ["automotive", "car", "vehicle", "bmw", "mercedes", "audi", "porsche", "bosch"],
            "software": ["software", "app", "platform", "saas", "technology", "digital", "cloud"],
            "manufacturing": ["manufacturing", "production", "factory", "industrial", "machinery"],
            "energy": ["energy", "power", "electricity", "renewable", "solar", "wind"],
            "telecommunications": ["telecom", "mobile", "internet", "broadband", "communication"],
            "healthcare": ["healthcare", "medical", "hospital", "pharmaceutical", "health"]
        }
    
    def _load_content_patterns(self) -> Dict[str, List[str]]:
        """Load content analysis patterns"""
        return {
            "gdpr_articles": [r"art\.\s*(\d+)", r"article\s*(\d+)", r"artikel\s*(\d+)"],
            "compliance_terms": ["einwilligung", "consent", "datenverarbeitung", "data processing", "aufsichtsbehörde"],
            "authority_references": ["bfdi", "baylda", "lfd", "ldi", "datenschutzbehörde"]
        }
    
    def _load_business_context_patterns(self) -> Dict[str, List[str]]:
        """Load business context patterns"""
        return {
            "company_size": {
                "small": ["klein", "startup", "gmbh", "einzelunternehmen"],
                "medium": ["mittelstand", "sme", "medium", "mittlere"],
                "large": ["konzern", "corporation", "ag", "large", "enterprise"]
            }
        }
    
    # Helper methods for content extraction
    def _combine_document_content(self, documents: List[Document]) -> str:
        """Combine all document content for analysis"""
        return " ".join(doc.content for doc in documents if hasattr(doc, 'content'))
    
    def _extract_geographic_indicators(self, content: str, suggested_state: Optional[str]) -> List[str]:
        """Extract geographic indicators from content"""
        indicators = []
        
        # Add suggested state if provided
        if suggested_state:
            indicators.append(suggested_state.lower())
        
        # Extract from content
        for region, patterns in self.geographic_patterns.items():
            for pattern in patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    indicators.append(region)
                    break
        
        return list(set(indicators))
    
    def _extract_industry_indicators(self, content: str, suggested_industry: Optional[str]) -> List[str]:
        """Extract industry indicators from content"""
        indicators = []
        
        # Add suggested industry if provided
        if suggested_industry:
            indicators.append(suggested_industry.lower())
        
        # Extract from content
        for industry, patterns in self.industry_patterns.items():
            for pattern in patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    indicators.append(industry)
                    break
        
        return list(set(indicators))
    
    def _extract_content_indicators(self, content: str) -> Dict[str, List[str]]:
        """Extract various content indicators"""
        indicators = {}
        
        # GDPR articles
        gdpr_articles = []
        for pattern in self.content_patterns["gdpr_articles"]:
            matches = re.findall(pattern, content, re.IGNORECASE)
            gdpr_articles.extend([f"Art. {match}" for match in matches])
        indicators["gdpr_articles"] = list(set(gdpr_articles))
        
        # Compliance terms
        compliance_terms = []
        for term in self.content_patterns["compliance_terms"]:
            if re.search(term, content, re.IGNORECASE):
                compliance_terms.append(term)
        indicators["compliance_terms"] = compliance_terms
        
        # Authority references
        authority_refs = []
        for ref in self.content_patterns["authority_references"]:
            if re.search(ref, content, re.IGNORECASE):
                authority_refs.append(ref)
        indicators["authority_references"] = authority_refs
        
        return indicators
    
    def _extract_industry_from_activity(self, activity: str) -> List[str]:
        """Extract industry indicators from business activity description"""
        industries = []
        activity_lower = activity.lower()
        
        for industry, patterns in self.industry_patterns.items():
            for pattern in patterns:
                if pattern in activity_lower:
                    industries.append(industry)
                    break
        
        return industries