# app/services/german_authority_engine/compliance_scorer.py
"""
Compliance Scorer

Calculates authority-specific compliance scores based on requirement
mappings and authority enforcement patterns. Provides detailed scoring
methodology aligned with German DPA enforcement approaches.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import structlog

from app.models.database import Document
from .authority_profiles import AuthorityProfile, AuthorityRequirement

logger = structlog.get_logger()

@dataclass
class ComplianceScore:
    """Detailed compliance score breakdown"""
    overall_score: float
    requirement_scores: Dict[str, float]
    weighted_score: float
    confidence_level: float
    
    # Categorized requirements
    requirements_met: List[str]
    requirements_partial: List[str]
    requirements_missing: List[str]
    
    # Score components
    technical_score: float
    documentation_score: float
    process_score: float
    industry_score: float

class ComplianceScorer:
    """
    Authority-specific compliance scoring engine
    
    Calculates compliance scores using German DPA-specific weighting
    and enforcement priority considerations.
    """
    
    def __init__(self):
        # Authority-specific scoring weights
        self.authority_weights = {
            "bfdi": {
                "international_transfers": 0.4,
                "documentation": 0.3,
                "technical_measures": 0.2,
                "process_implementation": 0.1
            },
            "baylda": {
                "technical_measures": 0.35,
                "industry_specific": 0.25,
                "documentation": 0.25,
                "process_implementation": 0.15
            },
            "lfd_bw": {
                "technical_measures": 0.4,
                "engineering_integration": 0.25,
                "documentation": 0.2,
                "process_implementation": 0.15
            },
            "default": {
                "documentation": 0.3,
                "technical_measures": 0.3,
                "process_implementation": 0.25,
                "industry_specific": 0.15
            }
        }
        
        # Enforcement priority multipliers
        self.priority_multipliers = {
            "high": 1.5,
            "medium": 1.0,
            "low": 0.7
        }
    
    async def calculate_authority_compliance(
        self,
        documents: List[Document],
        requirements: List[AuthorityRequirement],
        requirement_mappings: Dict[str, Any],
        authority_profile: AuthorityProfile
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive compliance score for specific authority
        
        Returns detailed scoring breakdown with authority-specific weighting
        """
        logger.info(
            "Calculating authority compliance score",
            authority=authority_profile.authority_id,
            requirements=len(requirements)
        )
        
        mappings = requirement_mappings.get("mappings", {})
        
        # Calculate individual requirement scores
        requirement_scores = {}
        weighted_scores = {}
        
        for requirement in requirements:
            article_ref = requirement.article_reference
            mapping = mappings.get(article_ref)
            
            if mapping:
                # Base score from requirement mapping
                base_score = mapping.coverage_score
                
                # Apply enforcement priority weighting
                priority_weight = self.priority_multipliers.get(
                    requirement.enforcement_priority, 1.0
                )
                
                # Apply authority-specific weighting
                authority_weight = self._get_authority_weight(
                    authority_profile, requirement
                )
                
                # Calculate final requirement score
                final_score = base_score * priority_weight * authority_weight
                
                requirement_scores[article_ref] = base_score
                weighted_scores[article_ref] = final_score
            else:
                requirement_scores[article_ref] = 0.0
                weighted_scores[article_ref] = 0.0
        
        # Calculate overall scores
        overall_score = self._calculate_overall_score(requirement_scores)
        weighted_score = self._calculate_weighted_score(weighted_scores)
        
        # Categorize requirements
        requirements_met = []
        requirements_partial = []
        requirements_missing = []
        
        for article_ref, score in requirement_scores.items():
            if score >= 0.8:
                requirements_met.append(article_ref)
            elif score >= 0.3:
                requirements_partial.append(article_ref)
            else:
                requirements_missing.append(article_ref)
        
        # Calculate component scores
        component_scores = await self._calculate_component_scores(
            documents, requirements, mappings, authority_profile
        )
        
        # Calculate confidence level
        confidence_level = self._calculate_confidence_level(
            mappings, requirement_scores
        )
        
        result = {
            "overall_score": overall_score,
            "weighted_score": weighted_score,
            "confidence_level": confidence_level,
            "requirement_scores": requirement_scores,
            "requirements_met": requirements_met,
            "requirements_partial": requirements_partial,
            "requirements_missing": requirements_missing,
            "component_scores": component_scores,
            "scoring_methodology": self._get_scoring_methodology(authority_profile)
        }
        
        logger.info(
            "Authority compliance scoring completed",
            authority=authority_profile.authority_id,
            overall_score=overall_score,
            weighted_score=weighted_score,
            requirements_met=len(requirements_met),
            requirements_missing=len(requirements_missing)
        )
        
        return result
    
    def _get_authority_weight(
        self, 
        authority_profile: AuthorityProfile, 
        requirement: AuthorityRequirement
    ) -> float:
        """Get authority-specific weighting for requirement"""
        
        authority_id = authority_profile.authority_id
        weights = self.authority_weights.get(authority_id, self.authority_weights["default"])
        
        # Map requirement to scoring category
        article_ref = requirement.article_reference
        
        if "transfer" in requirement.requirement_text.lower():
            return weights.get("international_transfers", 1.0)
        elif "technical" in requirement.requirement_text.lower():
            return weights.get("technical_measures", 1.0)
        elif "documentation" in requirement.requirement_text.lower():
            return weights.get("documentation", 1.0)
        elif any(industry in requirement.industry_specific for industry in authority_profile.industry_focus):
            return weights.get("industry_specific", 1.0)
        else:
            return weights.get("process_implementation", 1.0)
    
    def _calculate_overall_score(self, requirement_scores: Dict[str, float]) -> float:
        """Calculate overall compliance score"""
        if not requirement_scores:
            return 0.0
        
        return sum(requirement_scores.values()) / len(requirement_scores)
    
    def _calculate_weighted_score(self, weighted_scores: Dict[str, float]) -> float:
        """Calculate weighted compliance score"""
        if not weighted_scores:
            return 0.0
        
        return sum(weighted_scores.values()) / len(weighted_scores)
    
    async def _calculate_component_scores(
        self,
        documents: List[Document],
        requirements: List[AuthorityRequirement],
        mappings: Dict[str, Any],
        authority_profile: AuthorityProfile
    ) -> Dict[str, float]:
        """Calculate scores for different compliance components"""
        
        # Technical measures score
        technical_requirements = [
            req for req in requirements
            if "technical" in req.requirement_text.lower() or 
               req.article_reference in ["Art. 25", "Art. 32"]
        ]
        technical_score = self._calculate_category_score(technical_requirements, mappings)
        
        # Documentation score
        documentation_requirements = [
            req for req in requirements
            if req.article_reference in ["Art. 13", "Art. 14", "Art. 30"]
        ]
        documentation_score = self._calculate_category_score(documentation_requirements, mappings)
        
        # Process implementation score
        process_requirements = [
            req for req in requirements
            if req.article_reference in ["Art. 6", "Art. 7", "Art. 35"]
        ]
        process_score = self._calculate_category_score(process_requirements, mappings)
        
        # Industry-specific score
        industry_score = self._calculate_industry_score(
            requirements, mappings, authority_profile
        )
        
        return {
            "technical_score": technical_score,
            "documentation_score": documentation_score,
            "process_score": process_score,
            "industry_score": industry_score
        }
    
    def _calculate_category_score(
        self, 
        category_requirements: List[AuthorityRequirement], 
        mappings: Dict[str, Any]
    ) -> float:
        """Calculate score for specific requirement category"""
        if not category_requirements:
            return 0.0
        
        scores = []
        for req in category_requirements:
            mapping = mappings.get(req.article_reference)
            if mapping:
                scores.append(mapping.coverage_score)
            else:
                scores.append(0.0)
        
        return sum(scores) / len(scores)
    
    def _calculate_industry_score(
        self,
        requirements: List[AuthorityRequirement],
        mappings: Dict[str, Any],
        authority_profile: AuthorityProfile
    ) -> float:
        """Calculate industry-specific compliance score"""
        
        industry_requirements = [
            req for req in requirements
            if req.industry_specific and 
               any(industry in authority_profile.industry_focus 
                   for industry in req.industry_specific)
        ]
        
        if not industry_requirements:
            return 0.5  # Neutral score if no industry requirements
        
        return self._calculate_category_score(industry_requirements, mappings)
    
    def _calculate_confidence_level(
        self,
        mappings: Dict[str, Any],
        requirement_scores: Dict[str, float]
    ) -> float:
        """Calculate confidence level in scoring accuracy"""
        
        if not mappings:
            return 0.0
        
        confidence_scores = []
        for mapping in mappings.values():
            if hasattr(mapping, 'confidence_level'):
                confidence_scores.append(mapping.confidence_level)
        
        # Base confidence on mapping confidence and evidence strength
        base_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.5
        
        # Adjust based on evidence coverage
        evidence_coverage = sum(1 for score in requirement_scores.values() if score > 0.1) / len(requirement_scores)
        
        # Final confidence calculation
        final_confidence = (base_confidence * 0.7) + (evidence_coverage * 0.3)
        
        return min(1.0, final_confidence)
    
    def _get_scoring_methodology(self, authority_profile: AuthorityProfile) -> Dict[str, Any]:
        """Get scoring methodology explanation for transparency"""
        
        authority_id = authority_profile.authority_id
        weights = self.authority_weights.get(authority_id, self.authority_weights["default"])
        
        methodology = {
            "authority": authority_profile.name,
            "scoring_approach": f"Authority-specific weighting based on {authority_profile.enforcement_style}",
            "weighting_factors": weights,
            "priority_multipliers": self.priority_multipliers,
            "special_considerations": []
        }
        
        # Add authority-specific considerations
        if authority_id == "bfdi":
            methodology["special_considerations"].append(
                "Enhanced weighting for international transfer compliance"
            )
        elif authority_id in ["baylda", "lfd_bw"]:
            methodology["special_considerations"].append(
                "Increased emphasis on technical measures and industry-specific requirements"
            )
        
        # Add industry considerations
        if "automotive" in authority_profile.industry_focus:
            methodology["special_considerations"].append(
                "Automotive industry-specific compliance requirements emphasized"
            )
        
        return methodology
    
    async def calculate_comparative_score(
        self,
        base_scores: Dict[str, float],
        authority_profiles: List[AuthorityProfile]
    ) -> Dict[str, Any]:
        """Calculate comparative compliance scores across authorities"""
        
        comparative_analysis = {
            "authority_rankings": {},
            "score_variations": {},
            "risk_assessment": {},
            "optimization_recommendations": []
        }
        
        # Rank authorities by compliance score
        for profile in authority_profiles:
            authority_id = profile.authority_id
            if authority_id in base_scores:
                comparative_analysis["authority_rankings"][authority_id] = {
                    "score": base_scores[authority_id],
                    "enforcement_style": profile.enforcement_style,
                    "risk_level": self._assess_authority_risk_level(
                        base_scores[authority_id], profile
                    )
                }
        
        # Calculate score variations
        scores = list(base_scores.values())
        if scores:
            comparative_analysis["score_variations"] = {
                "highest_score": max(scores),
                "lowest_score": min(scores),
                "average_score": sum(scores) / len(scores),
                "score_range": max(scores) - min(scores)
            }
        
        # Generate optimization recommendations
        comparative_analysis["optimization_recommendations"] = self._generate_optimization_recommendations(
            base_scores, authority_profiles
        )
        
        return comparative_analysis
    
    def _assess_authority_risk_level(self, score: float, profile: AuthorityProfile) -> str:
        """Assess risk level for specific authority based on score and enforcement style"""
        
        # Base risk assessment
        if score >= 0.85:
            base_risk = "low"
        elif score >= 0.65:
            base_risk = "medium"
        elif score >= 0.45:
            base_risk = "high"
        else:
            base_risk = "critical"
        
        # Adjust based on enforcement style
        if "strict" in profile.enforcement_style.lower() and base_risk in ["medium", "high"]:
            risk_levels = {"medium": "high", "high": "critical"}
            return risk_levels.get(base_risk, base_risk)
        
        return base_risk
    
    def _generate_optimization_recommendations(
        self,
        scores: Dict[str, float],
        profiles: List[AuthorityProfile]
    ) -> List[str]:
        """Generate recommendations for compliance optimization"""
        
        recommendations = []
        
        # Find lowest scoring authority
        if scores:
            lowest_authority = min(scores, key=scores.get)
            lowest_score = scores[lowest_authority]
            
            if lowest_score < 0.7:
                recommendations.append(
                    f"Priority focus needed for {lowest_authority} compliance (score: {lowest_score:.2f})"
                )
        
        # Authority-specific recommendations
        for profile in profiles:
            authority_id = profile.authority_id
            if authority_id in scores and scores[authority_id] < 0.75:
                if "automotive" in profile.industry_focus:
                    recommendations.append(
                        f"Enhance automotive-specific privacy measures for {authority_id}"
                    )
                if "technical" in profile.enforcement_style.lower():
                    recommendations.append(
                        f"Strengthen technical security documentation for {authority_id}"
                    )
        
        return recommendations[:5]  # Limit to top 5 recommendations