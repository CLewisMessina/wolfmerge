# app/services/german_authority_engine/authority_analyzer.py
"""
German Authority Compliance Analyzer

Core engine for analyzing documents against specific German data protection
authority requirements. Provides authority-specific compliance scoring,
gap analysis, and enforcement risk assessment.
"""

import asyncio
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Union
from datetime import datetime, timezone
import structlog

from app.models.database import Document
from .authority_profiles import (
    GermanAuthority, AuthorityProfile, AuthorityRequirement,
    get_authority_profile, get_all_authorities
)
from .authority_detector import AuthorityDetector
from .requirement_mapper import RequirementMapper
from .compliance_scorer import ComplianceScorer

logger = structlog.get_logger()

@dataclass
class AuthorityComplianceAnalysis:
    """Complete compliance analysis for specific German authority"""
    authority_id: str
    authority_name: str
    jurisdiction: str
    
    # Compliance assessment
    compliance_score: float
    confidence_level: float
    
    # Requirement analysis
    requirements_met: List[str]
    requirements_missing: List[str]
    requirements_partial: List[str]
    
    # Risk assessment
    audit_readiness_score: float
    penalty_risk_level: str  # low, medium, high, critical
    estimated_penalty_range: str
    
    # Recommendations
    priority_actions: List[str]
    compliance_improvements: List[str]
    audit_preparation_steps: List[str]
    
    # Authority-specific insights
    enforcement_likelihood: float
    audit_focus_areas: List[str]
    industry_specific_guidance: List[str]
    
    # Processing metadata
    documents_analyzed: int
    analysis_timestamp: str
    processing_time_seconds: float

class GermanAuthorityEngine:
    """
    Main engine for German authority-specific compliance analysis
    
    Analyzes documents against specific German DPA requirements,
    calculates authority-specific compliance scores, and provides
    targeted recommendations for audit preparation.
    """
    
    def __init__(self):
        self.authority_detector = AuthorityDetector()
        self.requirement_mapper = RequirementMapper()
        self.compliance_scorer = ComplianceScorer()
        
        logger.info("German Authority Engine initialized with 16+ authority profiles")
    
    async def analyze_for_authority(
        self,
        documents: List[Document],
        authority: Union[str, GermanAuthority],
        industry: Optional[str] = None,
        include_audit_prep: bool = True
    ) -> AuthorityComplianceAnalysis:
        """
        Analyze documents for specific German authority compliance
        
        Args:
            documents: List of documents to analyze
            authority: German authority ID or enum
            industry: Optional industry context for targeted requirements
            include_audit_prep: Whether to include audit preparation guidance
            
        Returns:
            Comprehensive authority-specific compliance analysis
        """
        start_time = datetime.now(timezone.utc)
        
        # Get authority profile
        authority_profile = get_authority_profile(authority)
        if not authority_profile:
            raise ValueError(f"Unknown German authority: {authority}")
        
        logger.info(
            "Starting authority-specific analysis",
            authority=authority_profile.authority_id,
            documents=len(documents),
            industry=industry
        )
        
        try:
            # Filter requirements by industry if specified
            relevant_requirements = self._filter_requirements_by_industry(
                authority_profile.specific_requirements, industry
            )
            
            # Map document content to authority requirements
            requirement_mappings = await self.requirement_mapper.map_documents_to_requirements(
                documents, relevant_requirements, authority_profile
            )
            
            # Calculate compliance scores
            compliance_scores = await self.compliance_scorer.calculate_authority_compliance(
                documents, relevant_requirements, requirement_mappings, authority_profile
            )
            
            # Assess audit readiness
            audit_assessment = await self._assess_audit_readiness(
                documents, authority_profile, compliance_scores, requirement_mappings
            )
            
            # Generate recommendations
            recommendations = await self._generate_authority_recommendations(
                documents, authority_profile, compliance_scores, requirement_mappings, industry
            )
            
            # Calculate risk assessment
            risk_assessment = await self._calculate_risk_assessment(
                compliance_scores, authority_profile, requirement_mappings
            )
            
            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            
            # Compile comprehensive analysis
            analysis = AuthorityComplianceAnalysis(
                authority_id=authority_profile.authority_id,
                authority_name=authority_profile.name,
                jurisdiction=authority_profile.jurisdiction,
                
                compliance_score=compliance_scores["overall_score"],
                confidence_level=compliance_scores["confidence_level"],
                
                requirements_met=compliance_scores["requirements_met"],
                requirements_missing=compliance_scores["requirements_missing"],
                requirements_partial=compliance_scores["requirements_partial"],
                
                audit_readiness_score=audit_assessment["readiness_score"],
                penalty_risk_level=risk_assessment["risk_level"],
                estimated_penalty_range=risk_assessment["penalty_range"],
                
                priority_actions=recommendations["priority_actions"],
                compliance_improvements=recommendations["improvements"],
                audit_preparation_steps=recommendations["audit_prep"] if include_audit_prep else [],
                
                enforcement_likelihood=risk_assessment["enforcement_likelihood"],
                audit_focus_areas=audit_assessment["focus_areas"],
                industry_specific_guidance=recommendations["industry_guidance"],
                
                documents_analyzed=len(documents),
                analysis_timestamp=start_time.isoformat(),
                processing_time_seconds=processing_time
            )
            
            logger.info(
                "Authority analysis completed",
                authority=authority_profile.authority_id,
                compliance_score=analysis.compliance_score,
                risk_level=analysis.penalty_risk_level,
                processing_time=processing_time
            )
            
            return analysis
            
        except Exception as e:
            logger.error(
                "Authority analysis failed",
                authority=authority_profile.authority_id,
                error=str(e)
            )
            raise
    
    async def analyze_multi_authority(
        self,
        documents: List[Document],
        authorities: List[Union[str, GermanAuthority]],
        industry: Optional[str] = None
    ) -> Dict[str, AuthorityComplianceAnalysis]:
        """
        Analyze documents against multiple German authorities
        
        Useful for companies operating across multiple German states
        or needing federal + state authority compliance.
        """
        logger.info(
            "Starting multi-authority analysis",
            authorities=len(authorities),
            documents=len(documents)
        )
        
        # Run analyses in parallel for better performance
        analysis_tasks = [
            self.analyze_for_authority(documents, authority, industry)
            for authority in authorities
        ]
        
        results = await asyncio.gather(*analysis_tasks, return_exceptions=True)
        
        # Process results
        analyses = {}
        for i, result in enumerate(results):
            authority = authorities[i]
            authority_id = authority if isinstance(authority, str) else authority.value
            
            if isinstance(result, Exception):
                logger.error(
                    "Multi-authority analysis failed for authority",
                    authority=authority_id,
                    error=str(result)
                )
                continue
            
            analyses[authority_id] = result
        
        logger.info(
            "Multi-authority analysis completed",
            successful_analyses=len(analyses),
            total_authorities=len(authorities)
        )
        
        return analyses
    
    async def detect_relevant_authorities(
        self,
        documents: List[Document],
        industry: Optional[str] = None,
        location: Optional[str] = None
    ) -> List[GermanAuthority]:
        """
        Automatically detect which German authorities are most relevant
        based on document content, industry, and location.
        """
        return await self.authority_detector.detect_relevant_authorities(
            documents, industry, location
        )
    
    async def get_compliance_comparison(
        self,
        documents: List[Document],
        authorities: List[Union[str, GermanAuthority]],
        industry: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Compare compliance scores across multiple German authorities
        
        Helps identify which authority has the strictest requirements
        or poses the highest compliance risk.
        """
        analyses = await self.analyze_multi_authority(documents, authorities, industry)
        
        comparison = {
            "authority_scores": {},
            "highest_risk_authority": None,
            "lowest_compliance_authority": None,
            "average_compliance_score": 0.0,
            "unified_recommendations": [],
            "cross_authority_conflicts": []
        }
        
        if not analyses:
            return comparison
        
        # Extract scores and find extremes
        scores = {}
        risk_levels = {}
        
        for authority_id, analysis in analyses.items():
            scores[authority_id] = analysis.compliance_score
            risk_levels[authority_id] = analysis.penalty_risk_level
            
            comparison["authority_scores"][authority_id] = {
                "compliance_score": analysis.compliance_score,
                "risk_level": analysis.penalty_risk_level,
                "audit_readiness": analysis.audit_readiness_score
            }
        
        # Calculate comparisons
        comparison["lowest_compliance_authority"] = min(scores, key=scores.get)
        comparison["average_compliance_score"] = sum(scores.values()) / len(scores)
        
        # Find highest risk (prioritize critical > high > medium > low)
        risk_priority = {"critical": 4, "high": 3, "medium": 2, "low": 1}
        comparison["highest_risk_authority"] = max(
            risk_levels, 
            key=lambda x: risk_priority.get(risk_levels[x], 0)
        )
        
        # Generate unified recommendations
        all_recommendations = []
        for analysis in analyses.values():
            all_recommendations.extend(analysis.priority_actions)
        
        # Remove duplicates and prioritize
        comparison["unified_recommendations"] = list(dict.fromkeys(all_recommendations))[:10]
        
        # Identify potential conflicts (simplified for now)
        comparison["cross_authority_conflicts"] = await self._identify_cross_authority_conflicts(analyses)
        
        return comparison
    
    def _filter_requirements_by_industry(
        self,
        requirements: List[AuthorityRequirement],
        industry: Optional[str]
    ) -> List[AuthorityRequirement]:
        """Filter authority requirements by industry relevance"""
        if not industry:
            return requirements
        
        return [
            req for req in requirements
            if req.applies_to_industry(industry)
        ]
    
    async def _assess_audit_readiness(
        self,
        documents: List[Document],
        authority_profile: AuthorityProfile,
        compliance_scores: Dict[str, Any],
        requirement_mappings: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Assess readiness for authority audit"""
        
        # Base readiness on compliance score
        base_readiness = compliance_scores["overall_score"]
        
        # Adjust based on authority-specific factors
        audit_patterns = authority_profile.audit_patterns
        
        # Documentation readiness
        doc_readiness = self._assess_documentation_readiness(
            documents, authority_profile, requirement_mappings
        )
        
        # Process readiness
        process_readiness = self._assess_process_readiness(
            compliance_scores, authority_profile
        )
        
        # Final readiness score
        readiness_score = (base_readiness * 0.4 + doc_readiness * 0.3 + process_readiness * 0.3)
        
        # Identify focus areas based on authority patterns
        focus_areas = []
        if audit_patterns.get("technical_focus"):
            focus_areas.append("Technical security measures documentation")
        if audit_patterns.get("international_cooperation"):
            focus_areas.append("Cross-border transfer compliance")
        if "automotive" in authority_profile.industry_focus:
            focus_areas.append("Vehicle data processing compliance")
        
        return {
            "readiness_score": readiness_score,
            "documentation_readiness": doc_readiness,
            "process_readiness": process_readiness,
            "focus_areas": focus_areas,
            "estimated_audit_duration": audit_patterns.get("typical_duration", "2-4 months")
        }
    
    def _assess_documentation_readiness(
        self,
        documents: List[Document],
        authority_profile: AuthorityProfile,
        requirement_mappings: Dict[str, Any]
    ) -> float:
        """Assess documentation readiness for authority audit"""
        
        # Check for key document types
        doc_types_found = set()
        for doc in documents:
            if hasattr(doc, 'german_document_type') and doc.german_document_type:
                doc_types_found.add(doc.german_document_type)
        
        # Required documents based on authority requirements
        required_docs = {
            "datenschutzerklärung": 0.2,
            "verfahrensverzeichnis": 0.3,
            "dsfa": 0.25,
            "richtlinie": 0.15,
            "vertrag": 0.1
        }
        
        doc_score = sum(
            weight for doc_type, weight in required_docs.items()
            if doc_type in doc_types_found
        )
        
        return min(1.0, doc_score)
    
    def _assess_process_readiness(
        self,
        compliance_scores: Dict[str, Any],
        authority_profile: AuthorityProfile
    ) -> float:
        """Assess process readiness for authority audit"""
        
        # Base on requirements met vs missing
        requirements_met = len(compliance_scores.get("requirements_met", []))
        requirements_missing = len(compliance_scores.get("requirements_missing", []))
        total_requirements = requirements_met + requirements_missing
        
        if total_requirements == 0:
            return 0.5  # Neutral score if no requirements identified
        
        process_score = requirements_met / total_requirements
        
        # Adjust based on authority enforcement style
        if "collaborative" in authority_profile.enforcement_style.lower():
            process_score += 0.1  # Bonus for collaborative authorities
        
        return min(1.0, process_score)
    
    async def _generate_authority_recommendations(
        self,
        documents: List[Document],
        authority_profile: AuthorityProfile,
        compliance_scores: Dict[str, Any],
        requirement_mappings: Dict[str, Any],
        industry: Optional[str]
    ) -> Dict[str, List[str]]:
        """Generate authority-specific compliance recommendations"""
        
        priority_actions = []
        improvements = []
        audit_prep = []
        industry_guidance = []
        
        # Priority actions based on missing requirements
        for req_id in compliance_scores.get("requirements_missing", []):
            priority_actions.append(f"Implement {req_id} compliance measures")
        
        # Authority-specific improvements
        if authority_profile.authority_id == "baylda" and industry == "automotive":
            improvements.extend([
                "Implement automotive-specific privacy by design measures",
                "Enhance vehicle data processing documentation",
                "Establish connected car privacy controls"
            ])
        elif authority_profile.authority_id == "lfd_bw" and industry == "automotive":
            improvements.extend([
                "Integrate privacy into engineering system architecture",
                "Document technical security measures for automotive systems",
                "Establish engineering-grade privacy controls"
            ])
        elif authority_profile.authority_id == "bfdi":
            improvements.extend([
                "Enhance international transfer documentation",
                "Implement cross-border compliance coordination",
                "Establish federal-level audit readiness"
            ])
        
        # Audit preparation steps
        audit_prep.extend([
            f"Prepare documentation according to {authority_profile.name} standards",
            "Establish response team for authority inquiries",
            "Review and update compliance processes",
            "Prepare technical system documentation"
        ])
        
        # Industry-specific guidance
        if industry and industry in authority_profile.industry_focus:
            industry_guidance.extend([
                f"Follow {authority_profile.name} {industry} sector guidelines",
                f"Implement {industry}-specific privacy measures",
                f"Establish {industry} sector best practices"
            ])
        
        return {
            "priority_actions": priority_actions[:5],  # Top 5 priorities
            "improvements": improvements[:8],
            "audit_prep": audit_prep[:6],
            "industry_guidance": industry_guidance[:4]
        }
    
    async def _calculate_risk_assessment(
        self,
        compliance_scores: Dict[str, Any],
        authority_profile: AuthorityProfile,
        requirement_mappings: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate penalty and enforcement risk assessment"""
        
        base_compliance = compliance_scores["overall_score"]
        
        # Risk level based on compliance score
        if base_compliance >= 0.9:
            risk_level = "low"
            enforcement_likelihood = 0.1
        elif base_compliance >= 0.7:
            risk_level = "medium"
            enforcement_likelihood = 0.3
        elif base_compliance >= 0.5:
            risk_level = "high"
            enforcement_likelihood = 0.6
        else:
            risk_level = "critical"
            enforcement_likelihood = 0.85
        
        # Adjust based on authority enforcement patterns
        if "strict" in authority_profile.enforcement_style.lower():
            enforcement_likelihood += 0.2
        elif "collaborative" in authority_profile.enforcement_style.lower():
            enforcement_likelihood -= 0.1
        
        # Penalty range estimation
        high_priority_missing = sum(
            1 for req_id in compliance_scores.get("requirements_missing", [])
            if any(req.enforcement_priority == "high" 
                  for req in authority_profile.specific_requirements
                  if req.article_reference == req_id)
        )
        
        if high_priority_missing > 2:
            penalty_range = "€50,000 - €10,000,000"
        elif high_priority_missing > 0:
            penalty_range = "€10,000 - €1,000,000"
        else:
            penalty_range = "€1,000 - €50,000"
        
        return {
            "risk_level": risk_level,
            "enforcement_likelihood": min(0.95, enforcement_likelihood),
            "penalty_range": penalty_range,
            "high_priority_gaps": high_priority_missing
        }
    
    async def _identify_cross_authority_conflicts(
        self,
        analyses: Dict[str, AuthorityComplianceAnalysis]
    ) -> List[str]:
        """Identify potential conflicts between different authority requirements"""
        
        conflicts = []
        
        # Simple conflict detection - can be enhanced
        authority_requirements = {}
        for authority_id, analysis in analyses.items():
            authority_requirements[authority_id] = {
                "met": analysis.requirements_met,
                "missing": analysis.requirements_missing
            }
        
        # Check for conflicting requirements (simplified)
        if len(analyses) > 1:
            conflicts.append("Multiple authority jurisdictions may have varying requirements")
        
        return conflicts