# backend/app/services/german_authority_engine/integration/response_formatter.py
"""
Big 4 Response Formatter

Standardized response formatting for all Big 4 German Authority Engine endpoints.
Ensures consistent API responses that integrate seamlessly with existing
WolfMerge frontend and maintain compatibility with enhanced_compliance.py responses.

Key features:
- Consistent response structure across all Big 4 endpoints
- Enhanced metadata for UI intelligence
- Backwards compatibility with existing frontend
- Rich compliance insights and recommendations
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
import structlog

from ..big4.big4_profiles import Big4Authority, Big4AuthorityProfile
from ..big4.big4_analyzer import Big4ComplianceAnalysis, MultiAuthorityComparison
from ..big4.big4_detector import Big4DetectionResult
from ..big4.big4_templates import ComplianceTemplate, IndustryTemplate
from .multi_authority_analyzer import MultiAuthorityComparisonResult

logger = structlog.get_logger()

class Big4ResponseFormatter:
    """
    Standardized response formatter for Big 4 Authority Engine
    
    Formats all Big 4 responses to match WolfMerge API standards
    while providing enhanced German authority intelligence.
    """
    
    def __init__(self):
        self.response_version = "4.0.0"  # Big 4 enhancement version
        logger.info("Big 4 Response Formatter initialized")
    
    async def format_smart_detection_response(
        self,
        analysis_result: MultiAuthorityComparison,
        template_guidance: Optional[Dict[str, Any]],
        processing_time: float,
        workspace_id: str
    ) -> Dict[str, Any]:
        """Format smart detection and analysis response"""
        
        primary_analysis = analysis_result.primary_analysis
        comparative_analyses = analysis_result.comparative_analyses
        
        # Create enhanced compliance report
        compliance_report = {
            "framework": "gdpr_big4",
            "executive_summary": self._create_smart_detection_summary(
                primary_analysis, comparative_analyses, template_guidance
            ),
            "compliance_score": primary_analysis.compliance_score,
            "documents_analyzed": primary_analysis.documents_analyzed,
            "german_documents_detected": True,
            "priority_gaps": primary_analysis.priority_gaps[:5],
            "compliance_strengths": [
                f"Strong compliance with {primary_analysis.authority_name}",
                f"Audit readiness score: {primary_analysis.audit_readiness_score:.2f}",
                f"Big 4 authority optimization completed"
            ],
            "next_steps": primary_analysis.immediate_actions[:3],
            "big4_insights": {
                "primary_authority": {
                    "authority_id": primary_analysis.authority_id,
                    "authority_name": primary_analysis.authority_name,
                    "jurisdiction": primary_analysis.jurisdiction,
                    "enforcement_likelihood": primary_analysis.enforcement_likelihood,
                    "penalty_risk_level": primary_analysis.penalty_risk_level
                },
                "comparative_analysis": [
                    {
                        "authority_id": analysis.authority_id,
                        "authority_name": analysis.authority_name,
                        "compliance_score": analysis.compliance_score,
                        "audit_readiness": analysis.audit_readiness_score,
                        "penalty_risk": analysis.penalty_risk_level
                    }
                    for analysis in comparative_analyses
                ],
                "jurisdiction_advantages": analysis_result.jurisdiction_advantages,
                "conflicting_requirements": analysis_result.conflicting_requirements,
                "common_gaps": analysis_result.common_gaps
            }
        }
        
        # Create individual analyses (backwards compatibility)
        individual_analyses = [
            self._format_individual_analysis(primary_analysis, is_primary=True)
        ]
        
        for analysis in comparative_analyses:
            individual_analyses.append(
                self._format_individual_analysis(analysis, is_primary=False)
            )
        
        # Processing metadata with Big 4 enhancements
        processing_metadata = {
            "analysis_id": f"big4_{int(datetime.now(timezone.utc).timestamp())}",
            "processing_time": processing_time,
            "total_documents": primary_analysis.documents_analyzed,
            "framework": "gdpr_big4_smart_detection",
            "workspace_id": workspace_id,
            "big4_features": {
                "smart_authority_detection": True,
                "multi_authority_comparison": True,
                "industry_template_integration": template_guidance is not None,
                "jurisdiction_optimization": True,
                "enforcement_pattern_analysis": True
            },
            "detection_metadata": {
                "authorities_analyzed": len(individual_analyses),
                "primary_authority": primary_analysis.authority_id,
                "detection_confidence": 0.9,  # High confidence for Big 4 analysis
                "comparison_performed": len(comparative_analyses) > 0
            }
        }
        
        response = {
            "individual_analyses": individual_analyses,
            "compliance_report": compliance_report,
            "processing_metadata": processing_metadata
        }
        
        # Add template guidance if available
        if template_guidance:
            response["industry_template"] = template_guidance
        
        # Add strategic recommendations
        response["strategic_recommendations"] = {
            "recommended_primary_authority": analysis_result.recommended_primary_authority,
            "multi_jurisdiction_strategy": analysis_result.multi_jurisdiction_strategy,
            "cost_benefit_analysis": analysis_result.cost_benefit_analysis,
            "implementation_roadmap": self._format_implementation_roadmap(
                primary_analysis.immediate_actions,
                primary_analysis.compliance_improvements
            )
        }
        
        return response
    
    async def format_authority_analysis_response(
        self,
        analysis: Big4ComplianceAnalysis,
        template_guidance: Optional[Dict[str, Any]],
        processing_time: float,
        workspace_id: str
    ) -> Dict[str, Any]:
        """Format authority-specific analysis response"""
        
        # Create compliance report
        compliance_report = {
            "framework": f"gdpr_{analysis.authority_id}",
            "executive_summary": self._create_authority_summary(analysis, template_guidance),
            "compliance_score": analysis.compliance_score,
            "documents_analyzed": analysis.documents_analyzed,
            "german_documents_detected": True,
            "priority_gaps": analysis.priority_gaps,
            "compliance_strengths": [
                f"Authority-specific analysis for {analysis.authority_name}",
                f"Compliance score: {analysis.compliance_score:.2f}",
                f"Audit readiness: {analysis.audit_readiness_score:.2f}"
            ],
            "next_steps": analysis.immediate_actions,
            "authority_insights": {
                "enforcement_likelihood": analysis.enforcement_likelihood,
                "penalty_risk_level": analysis.penalty_risk_level,
                "estimated_penalty_range": analysis.estimated_penalty_range,
                "audit_preparation_steps": analysis.audit_preparation_steps,
                "industry_specific_guidance": analysis.industry_specific_guidance
            }
        }
        
        # Individual analysis (backwards compatibility)
        individual_analyses = [
            self._format_individual_analysis(analysis, is_primary=True)
        ]
        
        # Processing metadata
        processing_metadata = {
            "analysis_id": f"{analysis.authority_id}_{int(datetime.now(timezone.utc).timestamp())}",
            "processing_time": processing_time,
            "total_documents": analysis.documents_analyzed,
            "framework": f"gdpr_{analysis.authority_id}",
            "workspace_id": workspace_id,
            "big4_features": {
                "authority_specific_analysis": True,
                "enforcement_pattern_analysis": True,
                "industry_template_integration": template_guidance is not None,
                "penalty_risk_assessment": True,
                "audit_preparation_guidance": True
            },
            "authority_metadata": {
                "authority_id": analysis.authority_id,
                "authority_name": analysis.authority_name,
                "jurisdiction": analysis.jurisdiction,
                "analysis_confidence": analysis.confidence_level,
                "scoring_methodology": analysis.scoring_methodology
            }
        }
        
        response = {
            "individual_analyses": individual_analyses,
            "compliance_report": compliance_report,
            "processing_metadata": processing_metadata
        }
        
        # Add template guidance if available
        if template_guidance:
            response["industry_template"] = template_guidance
        
        return response
    
    async def format_comparison_response(
        self,
        comparison_result: MultiAuthorityComparisonResult,
        processing_time: float,
        workspace_id: str
    ) -> Dict[str, Any]:
        """Format multi-authority comparison response"""
        
        # Create compliance report for comparison
        compliance_report = {
            "framework": "gdpr_multi_authority_comparison",
            "executive_summary": self._create_comparison_summary(comparison_result),
            "compliance_score": max(analysis.compliance_score for analysis in comparison_result.all_analyses),
            "documents_analyzed": comparison_result.all_analyses[0].documents_analyzed,
            "german_documents_detected": True,
            "priority_gaps": comparison_result.common_gaps,
            "compliance_strengths": [
                f"Multi-authority analysis completed",
                f"Recommended primary: {comparison_result.primary_recommendation}",
                f"Authorities compared: {len(comparison_result.all_analyses)}"
            ],
            "next_steps": [
                f"Focus on {comparison_result.primary_recommendation} requirements",
                "Address common gaps across all jurisdictions",
                "Implement jurisdiction-specific optimizations"
            ],
            "comparison_insights": {
                "recommended_primary": comparison_result.primary_recommendation,
                "multi_authority_strategy": comparison_result.multi_authority_strategy,
                "common_gaps": comparison_result.common_gaps,
                "jurisdiction_specific_requirements": comparison_result.jurisdiction_specific_requirements,
                "pairwise_comparisons": [
                    {
                        "authority_a": comp.authority_a,
                        "authority_b": comp.authority_b,
                        "compliance_diff": comp.compliance_score_diff,
                        "jurisdiction_advantage": comp.jurisdiction_advantage
                    }
                    for comp in comparison_result.pairwise_comparisons
                ]
            }
        }
        
        # Individual analyses for all authorities
        individual_analyses = []
        for i, analysis in enumerate(comparison_result.all_analyses):
            individual_analyses.append(
                self._format_individual_analysis(analysis, is_primary=(i == 0))
            )
        
        # Processing metadata
        processing_metadata = {
            "analysis_id": f"comparison_{int(datetime.now(timezone.utc).timestamp())}",
            "processing_time": processing_time,
            "total_documents": comparison_result.all_analyses[0].documents_analyzed,
            "framework": "gdpr_multi_authority_comparison",
            "workspace_id": workspace_id,
            "big4_features": {
                "multi_authority_comparison": True,
                "pairwise_analysis": True,
                "jurisdiction_optimization": True,
                "cost_benefit_analysis": True,
                "implementation_roadmap": True
            },
            "comparison_metadata": {
                "authorities_compared": len(comparison_result.all_analyses),
                "recommended_primary": comparison_result.primary_recommendation,
                "strategy_type": "single" if not comparison_result.multi_authority_strategy else "multi",
                "optimization_potential": "high" if len(comparison_result.common_gaps) > 3 else "medium"
            }
        }
        
        response = {
            "individual_analyses": individual_analyses,
            "compliance_report": compliance_report,
            "processing_metadata": processing_metadata,
            "comparison_analysis": {
                "optimal_strategy": comparison_result.optimal_strategy,
                "implementation_roadmap": comparison_result.implementation_roadmap,
                "cost_estimates": comparison_result.compliance_cost_estimates,
                "risk_reduction_potential": comparison_result.risk_reduction_potential,
                "roi_analysis": comparison_result.roi_analysis
            }
        }
        
        return response
    
    async def format_detection_response(
        self,
        detection_result: Big4DetectionResult,
        template_recommendation: Optional[ComplianceTemplate]
    ) -> Dict[str, Any]:
        """Format authority detection response"""
        
        return {
            "detection_result": {
                "primary_authority": {
                    "authority_id": detection_result.primary_authority.value,
                    "confidence": detection_result.detection_confidence,
                    "reasoning": detection_result.all_scores[0].reasoning if detection_result.all_scores else []
                },
                "all_authorities": [
                    {
                        "authority_id": score.authority.value,
                        "total_score": score.total_score,
                        "confidence": score.confidence,
                        "reasoning": score.reasoning
                    }
                    for score in detection_result.all_scores
                ],
                "recommended_analysis_order": [auth.value for auth in detection_result.recommended_analysis_order],
                "multi_authority_needed": detection_result.multi_authority_needed,
                "fallback_suggestion": detection_result.fallback_suggestion
            },
            "geographic_indicators": detection_result.geographic_indicators,
            "industry_indicators": detection_result.industry_indicators,
            "content_indicators": detection_result.content_indicators,
            "template_recommendation": (
                {
                    "template_id": template_recommendation.template_id,
                    "industry": template_recommendation.industry.value,
                    "name": template_recommendation.name,
                    "description": template_recommendation.description,
                    "primary_authorities": [auth.value for auth in template_recommendation.primary_authorities],
                    "compliance_complexity": template_recommendation.compliance_complexity
                }
                if template_recommendation else None
            )
        }
    
    async def format_template_response(
        self,
        template: ComplianceTemplate,
        checklist: Optional[Dict[str, Any]],
        authority: Optional[Big4Authority]
    ) -> Dict[str, Any]:
        """Format industry template response"""
        
        response = {
            "template": {
                "template_id": template.template_id,
                "industry": template.industry.value,
                "name": template.name,
                "description": template.description,
                "compliance_complexity": template.compliance_complexity,
                "primary_authorities": [auth.value for auth in template.primary_authorities],
                "authority_priorities": template.authority_priorities,
                "required_documents": template.required_documents,
                "recommended_documents": template.recommended_documents,
                "critical_compliance_areas": template.critical_compliance_areas,
                "industry_specific_requirements": template.industry_specific_requirements,
                "typical_violations": template.typical_violations,
                "best_practices": template.best_practices,
                "high_risk_activities": template.high_risk_activities
            }
        }
        
        if checklist:
            response["authority_checklist"] = checklist
        
        if authority:
            response["authority_focus"] = {
                "authority_id": authority.value,
                "authority_specific_guidance": checklist.get("authority_specific", {}) if checklist else {}
            }
        
        return response
    
    # Helper methods for response formatting
    def _format_individual_analysis(
        self,
        analysis: Big4ComplianceAnalysis,
        is_primary: bool = False
    ) -> Dict[str, Any]:
        """Format individual analysis for backwards compatibility"""
        
        return {
            "filename": f"{analysis.authority_id}_analysis",
            "document_language": "de",
            "compliance_summary": self._create_individual_summary(analysis, is_primary),
            "control_mappings": [],
            "compliance_gaps": analysis.requirements_missing[:10],
            "risk_indicators": analysis.priority_gaps[:10],
            "german_insights": {
                "authority_id": analysis.authority_id,
                "authority_name": analysis.authority_name,
                "jurisdiction": analysis.jurisdiction,
                "enforcement_likelihood": analysis.enforcement_likelihood,
                "penalty_risk_level": analysis.penalty_risk_level,
                "estimated_penalty_range": analysis.estimated_penalty_range,
                "audit_readiness_score": analysis.audit_readiness_score,
                "compliance_score": analysis.compliance_score,
                "confidence_level": analysis.confidence_level,
                "requirements_met": analysis.requirements_met,
                "requirements_missing": analysis.requirements_missing,
                "immediate_actions": analysis.immediate_actions,
                "industry_guidance": analysis.industry_specific_guidance
            },
            "original_size": 0,
            "processing_time": analysis.processing_time_seconds
        }
    
    def _create_smart_detection_summary(
        self,
        primary_analysis: Big4ComplianceAnalysis,
        comparative_analyses: List[Big4ComplianceAnalysis],
        template_guidance: Optional[Dict[str, Any]]
    ) -> str:
        """Create executive summary for smart detection analysis"""
        
        summary_parts = [
            f"Big 4 Smart Detection Analysis: Primary authority identified as {primary_analysis.authority_name}.",
            f"Compliance score: {primary_analysis.compliance_score:.2f} with audit readiness: {primary_analysis.audit_readiness_score:.2f}.",
            f"Penalty risk level: {primary_analysis.penalty_risk_level}."
        ]
        
        if comparative_analyses:
            summary_parts.append(
                f"Comparative analysis with {len(comparative_analyses)} additional authorities completed."
            )
        
        if template_guidance:
            industry = template_guidance.get("industry", "unknown")
            summary_parts.append(f"Industry-specific guidance applied for {industry} sector.")
        
        summary_parts.extend([
            f"Priority action items: {len(primary_analysis.immediate_actions)}.",
            "Big 4 authority optimization and strategic recommendations provided."
        ])
        
        return " ".join(summary_parts)
    
    def _create_authority_summary(
        self,
        analysis: Big4ComplianceAnalysis,
        template_guidance: Optional[Dict[str, Any]]
    ) -> str:
        """Create executive summary for authority-specific analysis"""
        
        summary_parts = [
            f"Authority-Specific Analysis: {analysis.authority_name} compliance assessment completed.",
            f"Jurisdiction: {analysis.jurisdiction}.",
            f"Compliance score: {analysis.compliance_score:.2f} with {analysis.confidence_level:.2f} confidence.",
            f"Audit readiness: {analysis.audit_readiness_score:.2f}.",
            f"Penalty risk: {analysis.penalty_risk_level} ({analysis.estimated_penalty_range})."
        ]
        
        if template_guidance:
            industry = template_guidance.get("industry", "unknown")
            summary_parts.append(f"Industry-specific analysis for {industry} sector.")
        
        summary_parts.extend([
            f"Requirements met: {len(analysis.requirements_met)}.",
            f"Priority gaps identified: {len(analysis.priority_gaps)}.",
            f"Immediate actions required: {len(analysis.immediate_actions)}."
        ])
        
        return " ".join(summary_parts)
    
    def _create_comparison_summary(self, comparison_result: MultiAuthorityComparisonResult) -> str:
        """Create executive summary for multi-authority comparison"""
        
        summary_parts = [
            f"Multi-Authority Comparison: {len(comparison_result.all_analyses)} Big 4 authorities analyzed.",
            f"Recommended primary authority: {comparison_result.primary_recommendation}.",
            f"Common compliance gaps identified: {len(comparison_result.common_gaps)}."
        ]
        
        if comparison_result.multi_authority_strategy:
            summary_parts.append("Multi-authority strategy recommended for optimal compliance.")
        else:
            summary_parts.append("Single authority focus recommended for efficiency.")
        
        # Add best performing authority info
        best_analysis = max(comparison_result.all_analyses, key=lambda a: a.compliance_score)
        summary_parts.append(
            f"Highest compliance score: {best_analysis.compliance_score:.2f} ({best_analysis.authority_name})."
        )
        
        summary_parts.append("Strategic roadmap and cost-benefit analysis provided.")
        
        return " ".join(summary_parts)
    
    def _create_individual_summary(self, analysis: Big4ComplianceAnalysis, is_primary: bool) -> str:
        """Create individual analysis summary"""
        
        primary_indicator = "PRIMARY: " if is_primary else ""
        
        summary_parts = [
            f"{primary_indicator}{analysis.authority_name} compliance analysis.",
            f"Compliance score: {analysis.compliance_score:.2f}.",
            f"Audit readiness: {analysis.audit_readiness_score:.2f}.",
            f"Penalty risk: {analysis.penalty_risk_level}."
        ]
        
        if analysis.requirements_missing:
            summary_parts.append(f"Missing requirements: {len(analysis.requirements_missing)}.")
        
        if analysis.immediate_actions:
            summary_parts.append(f"Immediate actions: {len(analysis.immediate_actions)}.")
        
        return " ".join(summary_parts)
    
    def _format_implementation_roadmap(
        self,
        immediate_actions: List[str],
        compliance_improvements: List[str]
    ) -> List[Dict[str, Any]]:
        """Format implementation roadmap"""
        
        roadmap = []
        
        # Phase 1: Immediate actions
        if immediate_actions:
            roadmap.append({
                "phase": 1,
                "title": "Immediate Compliance Actions",
                "duration_weeks": 2,
                "priority": "critical",
                "actions": immediate_actions[:3],
                "description": "Address critical compliance gaps immediately"
            })
        
        # Phase 2: Compliance improvements
        if compliance_improvements:
            roadmap.append({
                "phase": 2,
                "title": "Compliance Enhancement",
                "duration_weeks": 6,
                "priority": "high",
                "actions": compliance_improvements[:5],
                "description": "Implement comprehensive compliance improvements"
            })
        
        # Phase 3: Ongoing monitoring
        roadmap.append({
            "phase": 3,
            "title": "Ongoing Monitoring & Optimization",
            "duration_weeks": 12,
            "priority": "medium",
            "actions": [
                "Establish regular compliance monitoring",
                "Conduct periodic compliance assessments",
                "Maintain authority relationship management"
            ],
            "description": "Ensure ongoing compliance and optimization"
        })
        
        return roadmap