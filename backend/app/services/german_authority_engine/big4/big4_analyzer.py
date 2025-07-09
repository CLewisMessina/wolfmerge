# backend/app/services/german_authority_engine/big4/big4_analyzer.py
"""
Big 4 Authority Analysis Engine

Specialized analysis engine for the Big 4 German authorities with:
- Authority-specific compliance scoring
- Enforcement pattern analysis
- Industry-specific requirements
- Penalty risk assessment
- Audit preparation guidance

Provides precise, actionable compliance analysis tailored to each authority's
enforcement style and priorities.
"""

import asyncio
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timezone
import structlog

from app.models.database import Document
from .big4_profiles import (
    Big4Authority, Big4AuthorityProfile, get_big4_authority_profile,
    get_authorities_by_industry, EnforcementProfile
)
from .big4_detector import Big4AuthorityDetector, Big4DetectionResult

logger = structlog.get_logger()

@dataclass
class Big4ComplianceAnalysis:
    """Comprehensive Big 4 authority compliance analysis"""
    authority_id: str
    authority_name: str
    jurisdiction: str
    
    # Compliance scoring
    compliance_score: float
    confidence_level: float
    scoring_methodology: str
    
    # Requirement analysis
    requirements_met: List[str]
    requirements_missing: List[str]
    requirements_partial: List[str]
    priority_gaps: List[str]
    
    # Authority-specific insights
    enforcement_likelihood: float
    penalty_risk_level: str
    estimated_penalty_range: str
    audit_readiness_score: float
    
    # Recommendations
    immediate_actions: List[str]
    compliance_improvements: List[str]
    audit_preparation_steps: List[str]
    industry_specific_guidance: List[str]
    
    # Analysis metadata
    documents_analyzed: int
    analysis_timestamp: str
    processing_time_seconds: float

@dataclass
class MultiAuthorityComparison:
    """Comparison analysis across multiple Big 4 authorities"""
    primary_analysis: Big4ComplianceAnalysis
    comparative_analyses: List[Big4ComplianceAnalysis]
    
    # Cross-authority insights
    conflicting_requirements: List[Dict[str, Any]]
    common_gaps: List[str]
    jurisdiction_advantages: Dict[str, List[str]]
    
    # Strategic recommendations
    recommended_primary_authority: str
    multi_jurisdiction_strategy: Optional[str]
    cost_benefit_analysis: Dict[str, Any]

class Big4ComplianceAnalyzer:
    """
    Advanced compliance analyzer for Big 4 German authorities
    
    Provides authority-specific analysis with deep understanding of
    enforcement patterns, industry requirements, and compliance priorities.
    """
    
    def __init__(self):
        self.detector = Big4AuthorityDetector()
        self.authority_weights = self._load_authority_scoring_weights()
        self.enforcement_patterns = self._load_enforcement_patterns()
        self.industry_requirements = self._load_industry_requirements()
        
        logger.info("Big 4 Compliance Analyzer initialized")
    
    async def analyze_for_authority(
        self,
        documents: List[Document],
        authority: Big4Authority,
        industry: Optional[str] = None,
        company_size: Optional[str] = None
    ) -> Big4ComplianceAnalysis:
        """
        Perform detailed compliance analysis for specific Big 4 authority
        
        Args:
            documents: Documents to analyze
            authority: Specific Big 4 authority
            industry: Industry context for targeted analysis
            company_size: Company size for SME-specific guidance
            
        Returns:
            Comprehensive authority-specific compliance analysis
        """
        
        start_time = datetime.now(timezone.utc)
        
        authority_profile = get_big4_authority_profile(authority)
        if not authority_profile:
            raise ValueError(f"Unknown Big 4 authority: {authority}")
        
        logger.info(
            "Starting Big 4 authority analysis",
            authority=authority.value,
            documents=len(documents),
            industry=industry,
            company_size=company_size
        )
        
        try:
            # Extract authority-specific requirements
            relevant_requirements = self._get_authority_requirements(
                authority_profile, industry, company_size
            )
            
            # Analyze documents against authority requirements
            compliance_analysis = await self._analyze_authority_compliance(
                documents, authority_profile, relevant_requirements, industry
            )
            
            # Calculate enforcement and penalty risk
            risk_assessment = await self._assess_enforcement_risk(
                compliance_analysis, authority_profile, industry, company_size
            )
            
            # Generate authority-specific recommendations
            recommendations = await self._generate_authority_recommendations(
                compliance_analysis, authority_profile, risk_assessment, industry
            )
            
            # Calculate audit readiness
            audit_readiness = self._calculate_audit_readiness(
                compliance_analysis, authority_profile, recommendations
            )
            
            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            
            analysis = Big4ComplianceAnalysis(
                authority_id=authority.value,
                authority_name=authority_profile.name,
                jurisdiction=authority_profile.jurisdiction,
                
                compliance_score=compliance_analysis["overall_score"],
                confidence_level=compliance_analysis["confidence"],
                scoring_methodology=compliance_analysis["methodology"],
                
                requirements_met=compliance_analysis["met"],
                requirements_missing=compliance_analysis["missing"],
                requirements_partial=compliance_analysis["partial"],
                priority_gaps=compliance_analysis["priority_gaps"],
                
                enforcement_likelihood=risk_assessment["enforcement_probability"],
                penalty_risk_level=risk_assessment["risk_level"],
                estimated_penalty_range=risk_assessment["penalty_estimate"],
                audit_readiness_score=audit_readiness,
                
                immediate_actions=recommendations["immediate"],
                compliance_improvements=recommendations["improvements"],
                audit_preparation_steps=recommendations["audit_prep"],
                industry_specific_guidance=recommendations["industry_guidance"],
                
                documents_analyzed=len(documents),
                analysis_timestamp=datetime.now(timezone.utc).isoformat(),
                processing_time_seconds=processing_time
            )
            
            logger.info(
                "Big 4 authority analysis completed",
                authority=authority.value,
                compliance_score=compliance_analysis["overall_score"],
                processing_time=processing_time
            )
            
            return analysis
            
        except Exception as e:
            logger.error(
                "Big 4 authority analysis failed",
                authority=authority.value,
                error=str(e)
            )
            raise
    
    async def analyze_with_smart_detection(
        self,
        documents: List[Document],
        industry: Optional[str] = None,
        company_location: Optional[str] = None,
        company_size: Optional[str] = None
    ) -> MultiAuthorityComparison:
        """
        Analyze with automatic authority detection and comparison
        
        Returns analysis for detected primary authority plus comparison
        with other relevant authorities.
        """
        
        logger.info(
            "Starting smart detection analysis",
            documents=len(documents),
            industry=industry,
            location=company_location
        )
        
        # Detect relevant authorities
        detection_result = await self.detector.detect_relevant_authorities(
            documents=documents,
            suggested_industry=industry,
            suggested_state=company_location,
            company_size=company_size
        )
        
        # Analyze for primary authority
        primary_analysis = await self.analyze_for_authority(
            documents=documents,
            authority=detection_result.primary_authority,
            industry=industry,
            company_size=company_size
        )
        
        # Analyze for comparison authorities (top 2 others)
        comparison_authorities = detection_result.recommended_analysis_order[1:3]
        comparative_analyses = []
        
        for authority in comparison_authorities:
            try:
                analysis = await self.analyze_for_authority(
                    documents=documents,
                    authority=authority,
                    industry=industry,
                    company_size=company_size
                )
                comparative_analyses.append(analysis)
            except Exception as e:
                logger.warning(
                    "Comparative analysis failed",
                    authority=authority.value,
                    error=str(e)
                )
        
        # Generate cross-authority insights
        conflicting_requirements = self._identify_conflicting_requirements(
            primary_analysis, comparative_analyses
        )
        
        common_gaps = self._identify_common_gaps(
            primary_analysis, comparative_analyses
        )
        
        jurisdiction_advantages = self._analyze_jurisdiction_advantages(
            primary_analysis, comparative_analyses
        )
        
        # Strategic recommendations
        strategic_recommendations = self._generate_strategic_recommendations(
            detection_result, primary_analysis, comparative_analyses
        )
        
        return MultiAuthorityComparison(
            primary_analysis=primary_analysis,
            comparative_analyses=comparative_analyses,
            conflicting_requirements=conflicting_requirements,
            common_gaps=common_gaps,
            jurisdiction_advantages=jurisdiction_advantages,
            recommended_primary_authority=strategic_recommendations["primary"],
            multi_jurisdiction_strategy=strategic_recommendations["multi_jurisdiction"],
            cost_benefit_analysis=strategic_recommendations["cost_benefit"]
        )
    
    async def _analyze_authority_compliance(
        self,
        documents: List[Document],
        authority_profile: Big4AuthorityProfile,
        requirements: List[Dict[str, Any]],
        industry: Optional[str]
    ) -> Dict[str, Any]:
        """Analyze compliance against authority-specific requirements"""
        
        # Extract content for analysis
        combined_content = self._extract_document_content(documents)
        
        # Score each requirement
        requirement_scores = {}
        met_requirements = []
        missing_requirements = []
        partial_requirements = []
        priority_gaps = []
        
        for requirement in requirements:
            score = await self._score_requirement_compliance(
                requirement, combined_content, authority_profile, industry
            )
            
            requirement_id = requirement["id"]
            requirement_scores[requirement_id] = score
            
            if score >= 0.8:
                met_requirements.append(requirement["description"])
            elif score >= 0.4:
                partial_requirements.append(requirement["description"])
            else:
                missing_requirements.append(requirement["description"])
                
                # Check if this is a priority requirement for this authority
                if requirement.get("priority") == "high":
                    priority_gaps.append(requirement["description"])
        
        # Calculate overall compliance score with authority weighting
        weighted_scores = []
        total_weight = 0
        
        for requirement in requirements:
            requirement_id = requirement["id"]
            weight = self._get_requirement_weight(requirement, authority_profile)
            score = requirement_scores[requirement_id]
            
            weighted_scores.append(score * weight)
            total_weight += weight
        
        overall_score = sum(weighted_scores) / max(total_weight, 1)
        
        # Calculate confidence based on document coverage
        confidence = self._calculate_analysis_confidence(
            documents, requirements, requirement_scores
        )
        
        return {
            "overall_score": overall_score,
            "confidence": confidence,
            "methodology": f"{authority_profile.authority_id}_weighted_scoring",
            "requirement_scores": requirement_scores,
            "met": met_requirements,
            "missing": missing_requirements,
            "partial": partial_requirements,
            "priority_gaps": priority_gaps
        }
    
    async def _score_requirement_compliance(
        self,
        requirement: Dict[str, Any],
        content: str,
        authority_profile: Big4AuthorityProfile,
        industry: Optional[str]
    ) -> float:
        """Score compliance for specific requirement (0-1)"""
        
        score = 0.0
        
        # Check for explicit compliance statements
        compliance_patterns = requirement.get("compliance_patterns", [])
        for pattern in compliance_patterns:
            if self._pattern_matches(pattern, content):
                score = max(score, 0.8)
        
        # Check for GDPR article references
        if requirement.get("gdpr_article"):
            article = requirement["gdpr_article"]
            if self._article_referenced(article, content):
                score = max(score, 0.6)
        
        # Industry-specific scoring adjustments
        if industry and industry in requirement.get("industries", []):
            score = min(1.0, score * 1.2)  # Boost for industry relevance
        
        # Authority-specific scoring adjustments
        if requirement.get("authority_priority", {}).get(authority_profile.authority_id):
            authority_multiplier = requirement["authority_priority"][authority_profile.authority_id]
            score = min(1.0, score * authority_multiplier)
        
        return score
    
    async def _assess_enforcement_risk(
        self,
        compliance_analysis: Dict[str, Any],
        authority_profile: Big4AuthorityProfile,
        industry: Optional[str],
        company_size: Optional[str]
    ) -> Dict[str, Any]:
        """Assess enforcement likelihood and penalty risk"""
        
        base_compliance = compliance_analysis["overall_score"]
        priority_gaps = len(compliance_analysis["priority_gaps"])
        
        # Calculate enforcement probability
        enforcement_factors = {
            "compliance_score": 1.0 - base_compliance,  # Lower compliance = higher risk
            "priority_gaps": min(1.0, priority_gaps * 0.2),  # Priority gaps increase risk
            "authority_style": self._get_authority_enforcement_factor(authority_profile),
            "industry_focus": self._get_industry_enforcement_factor(authority_profile, industry),
            "company_size": self._get_company_size_enforcement_factor(authority_profile, company_size)
        }
        
        # Weighted enforcement probability
        weights = {"compliance_score": 0.4, "priority_gaps": 0.3, "authority_style": 0.15, 
                  "industry_focus": 0.1, "company_size": 0.05}
        
        enforcement_probability = sum(
            enforcement_factors[factor] * weights[factor]
            for factor in weights
        )
        
        # Determine risk level
        if enforcement_probability > 0.7:
            risk_level = "critical"
        elif enforcement_probability > 0.5:
            risk_level = "high"
        elif enforcement_probability > 0.3:
            risk_level = "medium"
        else:
            risk_level = "low"
        
        # Estimate penalty range based on authority profile and compliance score
        penalty_estimate = self._estimate_penalty_range(
            authority_profile, base_compliance, priority_gaps, company_size
        )
        
        return {
            "enforcement_probability": enforcement_probability,
            "risk_level": risk_level,
            "penalty_estimate": penalty_estimate,
            "risk_factors": enforcement_factors
        }
    
    async def _generate_authority_recommendations(
        self,
        compliance_analysis: Dict[str, Any],
        authority_profile: Big4AuthorityProfile,
        risk_assessment: Dict[str, Any],
        industry: Optional[str]
    ) -> Dict[str, List[str]]:
        """Generate authority-specific recommendations"""
        
        recommendations = {
            "immediate": [],
            "improvements": [],
            "audit_prep": [],
            "industry_guidance": []
        }
        
        # Immediate actions for high-priority gaps
        priority_gaps = compliance_analysis["priority_gaps"]
        for gap in priority_gaps[:3]:  # Top 3 priority gaps
            action = self._generate_immediate_action(gap, authority_profile)
            if action:
                recommendations["immediate"].append(action)
        
        # Compliance improvements based on authority focus
        missing_requirements = compliance_analysis["missing"]
        for requirement in missing_requirements[:5]:  # Top 5 missing
            improvement = self._generate_improvement_recommendation(requirement, authority_profile)
            if improvement:
                recommendations["improvements"].append(improvement)
        
        # Audit preparation steps
        audit_steps = self._generate_audit_preparation_steps(
            authority_profile, compliance_analysis, risk_assessment
        )
        recommendations["audit_prep"] = audit_steps
        
        # Industry-specific guidance
        if industry:
            industry_guidance = self._generate_industry_guidance(
                authority_profile, industry, compliance_analysis
            )
            recommendations["industry_guidance"] = industry_guidance
        
        return recommendations
    
    # Helper methods for scoring and analysis
    def _get_authority_requirements(
        self,
        authority_profile: Big4AuthorityProfile,
        industry: Optional[str],
        company_size: Optional[str]
    ) -> List[Dict[str, Any]]:
        """Get authority-specific requirements"""
        
        base_requirements = self.industry_requirements.get("base", [])
        
        # Add authority-specific requirements
        authority_requirements = self.industry_requirements.get(
            authority_profile.authority_id.value, []
        )
        
        # Add industry-specific requirements if applicable
        industry_requirements = []
        if industry:
            industry_requirements = self.industry_requirements.get(
                f"{authority_profile.authority_id.value}_{industry}", []
            )
        
        # Combine and filter by company size if relevant
        all_requirements = base_requirements + authority_requirements + industry_requirements
        
        if company_size:
            all_requirements = [
                req for req in all_requirements
                if company_size in req.get("applicable_sizes", ["small", "medium", "large"])
            ]
        
        return all_requirements
    
    def _get_requirement_weight(
        self,
        requirement: Dict[str, Any],
        authority_profile: Big4AuthorityProfile
    ) -> float:
        """Get authority-specific weighting for requirement"""
        
        # Base weight
        weight = requirement.get("base_weight", 1.0)
        
        # Authority-specific multiplier
        authority_weights = self.authority_weights.get(authority_profile.authority_id.value, {})
        requirement_type = requirement.get("type", "general")
        
        multiplier = authority_weights.get(requirement_type, 1.0)
        
        return weight * multiplier
    
    def _calculate_analysis_confidence(
        self,
        documents: List[Document],
        requirements: List[Dict[str, Any]],
        requirement_scores: Dict[str, float]
    ) -> float:
        """Calculate confidence in analysis results"""
        
        # Base confidence from document count
        doc_confidence = min(1.0, len(documents) * 0.2)
        
        # Confidence from requirement coverage
        scored_requirements = len([s for s in requirement_scores.values() if s > 0.1])
        coverage_confidence = scored_requirements / max(len(requirements), 1)
        
        # Combined confidence
        return min(1.0, (doc_confidence + coverage_confidence) / 2)
    
    def _calculate_audit_readiness(
        self,
        compliance_analysis: Dict[str, Any],
        authority_profile: Big4AuthorityProfile,
        recommendations: Dict[str, List[str]]
    ) -> float:
        """Calculate audit readiness score (0-1)"""
        
        base_score = compliance_analysis["overall_score"]
        
        # Penalty for priority gaps
        priority_penalty = len(compliance_analysis["priority_gaps"]) * 0.1
        
        # Bonus for met requirements
        met_bonus = len(compliance_analysis["met"]) * 0.05
        
        # Authority-specific adjustments based on enforcement style
        if authority_profile.enforcement_profile.enforcement_style == "technical":
            # Technical authorities require higher compliance scores
            base_score *= 0.9
        elif authority_profile.enforcement_profile.enforcement_style == "education":
            # Education-first authorities are more lenient
            base_score = min(1.0, base_score * 1.1)
        
        readiness_score = max(0.0, min(1.0, base_score - priority_penalty + met_bonus))
        
        return readiness_score
    
    # Pattern matching and content analysis helpers
    def _extract_document_content(self, documents: List[Document]) -> str:
        """Extract and combine document content"""
        return " ".join(
            getattr(doc, 'content', '') for doc in documents
        )
    
    def _pattern_matches(self, pattern: str, content: str) -> bool:
        """Check if pattern matches in content"""
        import re
        try:
            return bool(re.search(pattern, content, re.IGNORECASE))
        except:
            return pattern.lower() in content.lower()
    
    def _article_referenced(self, article: str, content: str) -> bool:
        """Check if GDPR article is referenced in content"""
        import re
        patterns = [
            f"art\\.?\\s*{article}",
            f"article\\s*{article}",
            f"artikel\\s*{article}"
        ]
        
        for pattern in patterns:
            if re.search(pattern, content, re.IGNORECASE):
                return True
        return False
    
    # Load configuration data
    def _load_authority_scoring_weights(self) -> Dict[str, Dict[str, float]]:
        """Load authority-specific scoring weights"""
        return {
            "bfdi": {
                "data_protection_by_design": 1.2,
                "cross_border_transfers": 1.3,
                "data_subject_rights": 1.1,
                "breach_notification": 1.2,
                "general": 1.0
            },
            "baylda": {
                "consent_management": 1.3,
                "automotive_data": 1.4,
                "technical_measures": 1.2,
                "vendor_agreements": 1.2,
                "general": 1.0
            },
            "lfd_bw": {
                "data_protection_by_design": 1.4,
                "technical_measures": 1.3,
                "documentation": 1.2,
                "process_compliance": 1.3,
                "general": 1.0
            },
            "ldi_nrw": {
                "risk_assessment": 1.3,
                "manufacturing_compliance": 1.2,
                "data_minimization": 1.2,
                "security_measures": 1.1,
                "general": 1.0
            }
        }
    
    def _load_enforcement_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Load authority enforcement pattern data"""
        return {
            "bfdi": {
                "enforcement_likelihood": 0.6,
                "focus_areas": ["large_corporations", "cross_border", "federal_agencies"],
                "penalty_calculation": "revenue_based",
                "audit_trigger_threshold": 0.4
            },
            "baylda": {
                "enforcement_likelihood": 0.8,
                "focus_areas": ["automotive", "consent_violations", "sme_compliance"],
                "penalty_calculation": "fixed_plus_revenue",
                "audit_trigger_threshold": 0.5
            },
            "lfd_bw": {
                "enforcement_likelihood": 0.9,
                "focus_areas": ["technical_compliance", "software_companies", "documentation"],
                "penalty_calculation": "severity_based",
                "audit_trigger_threshold": 0.6
            },
            "ldi_nrw": {
                "enforcement_likelihood": 0.7,
                "focus_areas": ["manufacturing", "risk_management", "large_enterprises"],
                "penalty_calculation": "balanced",
                "audit_trigger_threshold": 0.4
            }
        }
    
    def _load_industry_requirements(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load industry and authority-specific requirements"""
        return {
            "base": [
                {
                    "id": "gdpr_art_5",
                    "description": "Data processing principles (Art. 5)",
                    "gdpr_article": "5",
                    "type": "data_protection_principles",
                    "priority": "high",
                    "base_weight": 1.2,
                    "compliance_patterns": ["lawfulness", "fairness", "transparency", "purpose limitation"],
                    "applicable_sizes": ["small", "medium", "large"]
                },
                {
                    "id": "gdpr_art_6",
                    "description": "Lawful basis for processing (Art. 6)",
                    "gdpr_article": "6",
                    "type": "legal_basis",
                    "priority": "high",
                    "base_weight": 1.3,
                    "compliance_patterns": ["legal basis", "consent", "contract", "legitimate interest"],
                    "applicable_sizes": ["small", "medium", "large"]
                },
                {
                    "id": "gdpr_art_13_14",
                    "description": "Information provided to data subjects (Art. 13-14)",
                    "gdpr_article": "13",
                    "type": "data_subject_rights",
                    "priority": "medium",
                    "base_weight": 1.1,
                    "compliance_patterns": ["privacy policy", "data subject information", "transparent information"],
                    "applicable_sizes": ["small", "medium", "large"]
                }
            ],
            "baylda_automotive": [
                {
                    "id": "automotive_consent",
                    "description": "Connected vehicle consent management",
                    "type": "consent_management",
                    "priority": "high",
                    "base_weight": 1.4,
                    "compliance_patterns": ["vehicle data consent", "telematics consent", "location tracking consent"],
                    "industries": ["automotive"],
                    "authority_priority": {"baylda": 1.5}
                },
                {
                    "id": "automotive_vendor",
                    "description": "Automotive supply chain data processing agreements",
                    "type": "vendor_agreements",
                    "priority": "high",
                    "base_weight": 1.3,
                    "compliance_patterns": ["supplier agreement", "data processing agreement", "automotive vendor"],
                    "industries": ["automotive"],
                    "authority_priority": {"baylda": 1.4}
                }
            ],
            "lfd_bw_software": [
                {
                    "id": "software_privacy_by_design",
                    "description": "Software development privacy by design",
                    "type": "data_protection_by_design",
                    "priority": "high",
                    "base_weight": 1.4,
                    "compliance_patterns": ["privacy by design", "software development", "system design"],
                    "industries": ["software"],
                    "authority_priority": {"lfd_bw": 1.5}
                },
                {
                    "id": "software_documentation",
                    "description": "Technical documentation requirements",
                    "type": "documentation",
                    "priority": "medium",
                    "base_weight": 1.2,
                    "compliance_patterns": ["technical documentation", "system documentation", "process documentation"],
                    "industries": ["software"],
                    "authority_priority": {"lfd_bw": 1.3}
                }
            ],
            "ldi_nrw_manufacturing": [
                {
                    "id": "manufacturing_risk_assessment",
                    "description": "Manufacturing process risk assessment",
                    "type": "risk_assessment",
                    "priority": "high",
                    "base_weight": 1.3,
                    "compliance_patterns": ["risk assessment", "manufacturing risk", "process risk"],
                    "industries": ["manufacturing"],
                    "authority_priority": {"ldi_nrw": 1.4}
                },
                {
                    "id": "manufacturing_security",
                    "description": "Manufacturing security measures",
                    "type": "security_measures",
                    "priority": "medium",
                    "base_weight": 1.2,
                    "compliance_patterns": ["security measures", "manufacturing security", "operational security"],
                    "industries": ["manufacturing"],
                    "authority_priority": {"ldi_nrw": 1.3}
                }
            ]
        }
    
    # Risk assessment helpers
    def _get_authority_enforcement_factor(self, authority_profile: Big4AuthorityProfile) -> float:
        """Get enforcement factor based on authority characteristics"""
        enforcement_data = self.enforcement_patterns.get(authority_profile.authority_id.value, {})
        return enforcement_data.get("enforcement_likelihood", 0.5)
    
    def _get_industry_enforcement_factor(
        self, 
        authority_profile: Big4AuthorityProfile, 
        industry: Optional[str]
    ) -> float:
        """Get enforcement factor based on industry focus"""
        if not industry:
            return 0.0
        
        enforcement_data = self.enforcement_patterns.get(authority_profile.authority_id.value, {})
        focus_areas = enforcement_data.get("focus_areas", [])
        
        # Higher enforcement likelihood if industry is in focus areas
        return 0.3 if industry in focus_areas else 0.0
    
    def _get_company_size_enforcement_factor(
        self,
        authority_profile: Big4AuthorityProfile,
        company_size: Optional[str]
    ) -> float:
        """Get enforcement factor based on company size"""
        if not company_size:
            return 0.0
        
        # SME-focused authorities more likely to audit SMEs
        if authority_profile.sme_focus and company_size in ["small", "medium"]:
            return 0.2
        elif not authority_profile.sme_focus and company_size == "large":
            return 0.3
        
        return 0.0
    
    def _estimate_penalty_range(
        self,
        authority_profile: Big4AuthorityProfile,
        compliance_score: float,
        priority_gaps: int,
        company_size: Optional[str]
    ) -> str:
        """Estimate penalty range based on authority and violation severity"""
        
        base_penalty = authority_profile.enforcement_profile.avg_penalty_amount
        
        # Parse base penalty amount (e.g., "€150,000")
        import re
        base_amount_match = re.search(r'€([\d,]+)', base_penalty)
        if base_amount_match:
            base_amount = int(base_amount_match.group(1).replace(',', ''))
        else:
            base_amount = 100000  # Default fallback
        
        # Calculate severity multiplier
        severity_multiplier = 1.0
        if compliance_score < 0.3:  # Very poor compliance
            severity_multiplier = 2.0
        elif compliance_score < 0.5:  # Poor compliance
            severity_multiplier = 1.5
        elif compliance_score < 0.7:  # Moderate compliance
            severity_multiplier = 1.2
        
        # Priority gaps increase penalty
        gap_multiplier = 1.0 + (priority_gaps * 0.2)
        
        # Company size adjustment
        size_multiplier = 1.0
        if company_size == "large":
            size_multiplier = 1.5
        elif company_size == "small":
            size_multiplier = 0.7
        
        # Calculate final range
        final_multiplier = severity_multiplier * gap_multiplier * size_multiplier
        estimated_penalty = int(base_amount * final_multiplier)
        
        # Create range (±30%)
        lower_bound = int(estimated_penalty * 0.7)
        upper_bound = int(estimated_penalty * 1.3)
        
        return f"€{lower_bound:,} - €{upper_bound:,}"
    
    # Recommendation generation helpers
    def _generate_immediate_action(self, gap: str, authority_profile: Big4AuthorityProfile) -> str:
        """Generate immediate action for priority gap"""
        action_templates = {
            "consent": f"Implement compliant consent management system following {authority_profile.name} guidance",
            "documentation": f"Create missing documentation according to {authority_profile.jurisdiction} requirements",
            "technical": f"Implement technical measures per {authority_profile.name} technical guidelines",
            "vendor": f"Update vendor agreements to meet {authority_profile.jurisdiction} standards"
        }
        
        for keyword, template in action_templates.items():
            if keyword.lower() in gap.lower():
                return template
        
        return f"Address compliance gap: {gap}"
    
    def _generate_improvement_recommendation(self, requirement: str, authority_profile: Big4AuthorityProfile) -> str:
        """Generate improvement recommendation for missing requirement"""
        return f"Implement {requirement} following {authority_profile.name} best practices"
    
    def _generate_audit_preparation_steps(
        self,
        authority_profile: Big4AuthorityProfile,
        compliance_analysis: Dict[str, Any],
        risk_assessment: Dict[str, Any]
    ) -> List[str]:
        """Generate audit preparation steps"""
        steps = []
        
        # Authority-specific preparation
        if authority_profile.authority_id == Big4Authority.BAYLDA:
            steps.extend([
                "Prepare automotive-specific consent documentation",
                "Review supplier data processing agreements",
                "Document technical measures for vehicle data protection"
            ])
        elif authority_profile.authority_id == Big4Authority.LFD_BW:
            steps.extend([
                "Prepare comprehensive technical documentation",
                "Document privacy by design implementation",
                "Review software development compliance processes"
            ])
        elif authority_profile.authority_id == Big4Authority.LDI_NRW:
            steps.extend([
                "Prepare manufacturing risk assessments",
                "Document operational security measures",
                "Review industrial data processing procedures"
            ])
        else:  # BfDI
            steps.extend([
                "Prepare cross-border transfer documentation",
                "Review federal compliance requirements",
                "Document enterprise-level security measures"
            ])
        
        # Risk-based additional steps
        if risk_assessment["enforcement_probability"] > 0.6:
            steps.append("Consider legal counsel consultation")
            steps.append("Prepare detailed compliance timeline")
        
        return steps
    
    def _generate_industry_guidance(
        self,
        authority_profile: Big4AuthorityProfile,
        industry: str,
        compliance_analysis: Dict[str, Any]
    ) -> List[str]:
        """Generate industry-specific guidance"""
        guidance = []
        
        industry_guidance_map = {
            "automotive": [
                f"Follow {authority_profile.name} automotive data protection guidelines",
                "Implement connected vehicle privacy controls",
                "Ensure telematics data minimization"
            ],
            "software": [
                f"Apply {authority_profile.name} software development standards",
                "Implement privacy by design in development lifecycle",
                "Document API data protection measures"
            ],
            "manufacturing": [
                f"Follow {authority_profile.name} industrial compliance guidelines",
                "Implement IoT device data protection",
                "Document manufacturing process privacy controls"
            ]
        }
        
        return industry_guidance_map.get(industry, [
            f"Follow general {authority_profile.name} compliance guidelines for {industry} sector"
        ])
    
    # Multi-authority comparison helpers
    def _identify_conflicting_requirements(
        self,
        primary_analysis: Big4ComplianceAnalysis,
        comparative_analyses: List[Big4ComplianceAnalysis]
    ) -> List[Dict[str, Any]]:
        """Identify conflicting requirements between authorities"""
        conflicts = []
        
        # Compare requirement sets (simplified implementation)
        primary_reqs = set(primary_analysis.requirements_met + primary_analysis.requirements_missing)
        
        for analysis in comparative_analyses:
            comp_reqs = set(analysis.requirements_met + analysis.requirements_missing)
            
            # Find requirements that exist in one but not the other
            unique_to_primary = primary_reqs - comp_reqs
            unique_to_comparative = comp_reqs - primary_reqs
            
            if unique_to_primary or unique_to_comparative:
                conflicts.append({
                    "primary_authority": primary_analysis.authority_id,
                    "comparative_authority": analysis.authority_id,
                    "unique_to_primary": list(unique_to_primary),
                    "unique_to_comparative": list(unique_to_comparative)
                })
        
        return conflicts
    
    def _identify_common_gaps(
        self,
        primary_analysis: Big4ComplianceAnalysis,
        comparative_analyses: List[Big4ComplianceAnalysis]
    ) -> List[str]:
        """Identify gaps common across all analyses"""
        if not comparative_analyses:
            return primary_analysis.requirements_missing
        
        # Start with primary analysis gaps
        common_gaps = set(primary_analysis.requirements_missing)
        
        # Find intersection with all comparative analyses
        for analysis in comparative_analyses:
            common_gaps = common_gaps.intersection(set(analysis.requirements_missing))
        
        return list(common_gaps)
    
    def _analyze_jurisdiction_advantages(
        self,
        primary_analysis: Big4ComplianceAnalysis,
        comparative_analyses: List[Big4ComplianceAnalysis]
    ) -> Dict[str, List[str]]:
        """Analyze advantages of each jurisdiction"""
        advantages = {}
        
        # Primary authority advantages
        advantages[primary_analysis.authority_id] = [
            f"Compliance score: {primary_analysis.compliance_score:.2f}",
            f"Audit readiness: {primary_analysis.audit_readiness_score:.2f}",
            f"Penalty risk: {primary_analysis.penalty_risk_level}"
        ]
        
        # Comparative authority advantages
        for analysis in comparative_analyses:
            advantages[analysis.authority_id] = [
                f"Compliance score: {analysis.compliance_score:.2f}",
                f"Audit readiness: {analysis.audit_readiness_score:.2f}",
                f"Penalty risk: {analysis.penalty_risk_level}"
            ]
        
        return advantages
    
    def _generate_strategic_recommendations(
        self,
        detection_result: Big4DetectionResult,
        primary_analysis: Big4ComplianceAnalysis,
        comparative_analyses: List[Big4ComplianceAnalysis]
    ) -> Dict[str, Any]:
        """Generate strategic recommendations for multi-authority context"""
        
        # Determine recommended primary authority
        recommended_primary = primary_analysis.authority_id
        
        # Check if another authority might be better
        for analysis in comparative_analyses:
            if (analysis.compliance_score > primary_analysis.compliance_score and
                analysis.audit_readiness_score > primary_analysis.audit_readiness_score):
                recommended_primary = analysis.authority_id
                break
        
        # Multi-jurisdiction strategy
        multi_jurisdiction = None
        if detection_result.multi_authority_needed:
            multi_jurisdiction = "Consider compliance optimization for multiple jurisdictions"
        
        # Cost-benefit analysis
        cost_benefit = {
            "recommended_investment": "Focus on highest-scoring authority requirements",
            "risk_reduction": f"Addressing common gaps provides maximum risk reduction",
            "efficiency_gain": "Single authority focus reduces complexity"
        }
        
        return {
            "primary": recommended_primary,
            "multi_jurisdiction": multi_jurisdiction,
            "cost_benefit": cost_benefit
        }