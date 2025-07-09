# backend/app/services/german_authority_engine/integration/multi_authority_analyzer.py
"""
Multi-Authority Comparison Analyzer

Advanced engine for comparing compliance analysis across multiple Big 4 authorities.
Provides strategic insights for jurisdiction selection, requirement conflicts,
and optimization strategies.

Key features:
- Side-by-side authority comparison
- Requirement conflict detection
- Jurisdiction optimization recommendations
- Cost-benefit analysis for compliance strategies
"""

import asyncio
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timezone
import structlog

from app.models.database import Document
from ..big4.big4_profiles import Big4Authority, Big4AuthorityProfile, get_big4_authority_profile
from ..big4.big4_analyzer import Big4ComplianceAnalyzer, Big4ComplianceAnalysis

logger = structlog.get_logger()

@dataclass
class AuthorityComparisonMetrics:
    """Comparison metrics between authorities"""
    authority_a: str
    authority_b: str
    
    # Compliance comparison
    compliance_score_diff: float
    audit_readiness_diff: float
    penalty_risk_comparison: str
    
    # Requirement differences
    unique_requirements_a: List[str]
    unique_requirements_b: List[str]
    conflicting_requirements: List[Dict[str, Any]]
    
    # Strategic insights
    jurisdiction_advantage: str  # which authority is more favorable
    complexity_difference: str   # implementation complexity comparison
    cost_benefit_ratio: float   # relative cost/benefit

@dataclass
class MultiAuthorityComparisonResult:
    """Complete multi-authority comparison analysis"""
    primary_recommendation: str
    all_analyses: List[Big4ComplianceAnalysis]
    
    # Comparison insights
    pairwise_comparisons: List[AuthorityComparisonMetrics]
    optimal_strategy: Dict[str, Any]
    
    # Cross-authority analysis
    common_gaps: List[str]
    universal_requirements: List[str]
    jurisdiction_specific_requirements: Dict[str, List[str]]
    
    # Strategic recommendations
    single_authority_recommendation: str
    multi_authority_strategy: Optional[str]
    implementation_roadmap: List[Dict[str, Any]]
    
    # Cost analysis
    compliance_cost_estimates: Dict[str, str]
    risk_reduction_potential: Dict[str, float]
    roi_analysis: Dict[str, Any]

class MultiAuthorityAnalyzer:
    """
    Advanced analyzer for multi-authority compliance comparison
    
    Provides strategic analysis for companies operating across multiple
    German jurisdictions or considering jurisdiction optimization.
    """
    
    def __init__(self):
        self.analyzer = Big4ComplianceAnalyzer()
        self.comparison_weights = self._load_comparison_weights()
        self.jurisdiction_costs = self._load_jurisdiction_cost_data()
        
        logger.info("Multi-Authority Analyzer initialized")
    
    async def compare_authorities(
        self,
        documents: List[Document],
        authorities: List[Big4Authority],
        industry: Optional[str] = None,
        company_size: Optional[str] = None
    ) -> MultiAuthorityComparisonResult:
        """
        Compare compliance analysis across multiple Big 4 authorities
        
        Args:
            documents: Documents to analyze
            authorities: List of authorities to compare (2-4 recommended)
            industry: Industry context for targeted analysis
            company_size: Company size for cost calculations
            
        Returns:
            Comprehensive multi-authority comparison with strategic recommendations
        """
        
        logger.info(
            "Starting multi-authority comparison",
            authorities=[auth.value for auth in authorities],
            documents=len(documents),
            industry=industry
        )
        
        if len(authorities) < 2:
            raise ValueError("At least 2 authorities required for comparison")
        
        if len(authorities) > 4:
            logger.warning("More than 4 authorities may result in complex analysis")
        
        try:
            # Perform analysis for each authority in parallel
            analysis_tasks = [
                self.analyzer.analyze_for_authority(
                    documents=documents,
                    authority=authority,
                    industry=industry,
                    company_size=company_size
                )
                for authority in authorities
            ]
            
            analyses = await asyncio.gather(*analysis_tasks)
            
            # Perform pairwise comparisons
            pairwise_comparisons = self._perform_pairwise_comparisons(analyses)
            
            # Identify cross-authority patterns
            common_gaps = self._identify_common_gaps(analyses)
            universal_requirements = self._identify_universal_requirements(analyses)
            jurisdiction_requirements = self._identify_jurisdiction_specific_requirements(analyses)
            
            # Generate optimal strategy
            optimal_strategy = await self._generate_optimal_strategy(
                analyses, pairwise_comparisons, industry, company_size
            )
            
            # Create implementation roadmap
            implementation_roadmap = self._create_implementation_roadmap(
                analyses, optimal_strategy, common_gaps
            )
            
            # Calculate cost analysis
            cost_estimates = self._calculate_compliance_costs(analyses, company_size)
            risk_reduction = self._calculate_risk_reduction_potential(analyses)
            roi_analysis = self._calculate_roi_analysis(cost_estimates, risk_reduction)
            
            # Determine primary recommendation
            primary_recommendation = self._determine_primary_recommendation(
                analyses, pairwise_comparisons, optimal_strategy
            )
            
            # Multi-authority strategy
            multi_authority_strategy = self._evaluate_multi_authority_strategy(
                analyses, pairwise_comparisons, cost_estimates
            )
            
            result = MultiAuthorityComparisonResult(
                primary_recommendation=primary_recommendation,
                all_analyses=analyses,
                pairwise_comparisons=pairwise_comparisons,
                optimal_strategy=optimal_strategy,
                common_gaps=common_gaps,
                universal_requirements=universal_requirements,
                jurisdiction_specific_requirements=jurisdiction_requirements,
                single_authority_recommendation=primary_recommendation,
                multi_authority_strategy=multi_authority_strategy,
                implementation_roadmap=implementation_roadmap,
                compliance_cost_estimates=cost_estimates,
                risk_reduction_potential=risk_reduction,
                roi_analysis=roi_analysis
            )
            
            logger.info(
                "Multi-authority comparison completed",
                primary_recommendation=primary_recommendation,
                authorities_compared=len(analyses),
                common_gaps=len(common_gaps)
            )
            
            return result
            
        except Exception as e:
            logger.error(
                "Multi-authority comparison failed",
                authorities=[auth.value for auth in authorities],
                error=str(e)
            )
            raise
    
    async def analyze_jurisdiction_optimization(
        self,
        documents: List[Document],
        current_authority: Big4Authority,
        industry: Optional[str] = None,
        company_size: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze potential benefits of switching jurisdictions
        
        Compares current authority with all other Big 4 authorities
        to identify optimization opportunities.
        """
        
        logger.info(
            "Starting jurisdiction optimization analysis",
            current_authority=current_authority.value,
            industry=industry
        )
        
        # Get all other Big 4 authorities
        all_authorities = list(Big4Authority)
        alternative_authorities = [auth for auth in all_authorities if auth != current_authority]
        
        # Compare current with alternatives
        comparison_authorities = [current_authority] + alternative_authorities
        comparison_result = await self.compare_authorities(
            documents=documents,
            authorities=comparison_authorities,
            industry=industry,
            company_size=company_size
        )
        
        # Calculate optimization potential
        current_analysis = comparison_result.all_analyses[0]  # First is current authority
        alternative_analyses = comparison_result.all_analyses[1:]  # Rest are alternatives
        
        optimization_opportunities = []
        for analysis in alternative_analyses:
            if (analysis.compliance_score > current_analysis.compliance_score or
                analysis.audit_readiness_score > current_analysis.audit_readiness_score):
                
                improvement_potential = {
                    "alternative_authority": analysis.authority_id,
                    "authority_name": analysis.authority_name,
                    "compliance_improvement": analysis.compliance_score - current_analysis.compliance_score,
                    "audit_readiness_improvement": analysis.audit_readiness_score - current_analysis.audit_readiness_score,
                    "penalty_risk_change": self._compare_penalty_risk(current_analysis, analysis),
                    "implementation_effort": self._estimate_switching_effort(current_analysis, analysis),
                    "cost_benefit_ratio": self._calculate_switching_cost_benefit(current_analysis, analysis)
                }
                optimization_opportunities.append(improvement_potential)
        
        # Sort by potential benefit
        optimization_opportunities.sort(key=lambda x: x["cost_benefit_ratio"], reverse=True)
        
        return {
            "current_authority": {
                "authority_id": current_authority.value,
                "compliance_score": current_analysis.compliance_score,
                "audit_readiness": current_analysis.audit_readiness_score,
                "penalty_risk": current_analysis.penalty_risk_level
            },
            "optimization_opportunities": optimization_opportunities,
            "recommendation": (
                "Consider jurisdiction optimization" if optimization_opportunities 
                else "Current jurisdiction appears optimal"
            ),
            "full_comparison": comparison_result
        }
    
    def _perform_pairwise_comparisons(
        self, 
        analyses: List[Big4ComplianceAnalysis]
    ) -> List[AuthorityComparisonMetrics]:
        """Perform pairwise comparisons between all authorities"""
        
        comparisons = []
        
        for i in range(len(analyses)):
            for j in range(i + 1, len(analyses)):
                analysis_a = analyses[i]
                analysis_b = analyses[j]
                
                comparison = self._compare_two_authorities(analysis_a, analysis_b)
                comparisons.append(comparison)
        
        return comparisons
    
    def _compare_two_authorities(
        self,
        analysis_a: Big4ComplianceAnalysis,
        analysis_b: Big4ComplianceAnalysis
    ) -> AuthorityComparisonMetrics:
        """Compare two authority analyses"""
        
        # Calculate score differences
        compliance_diff = analysis_b.compliance_score - analysis_a.compliance_score
        audit_readiness_diff = analysis_b.audit_readiness_score - analysis_a.audit_readiness_score
        
        # Compare penalty risks
        penalty_comparison = self._compare_penalty_risk(analysis_a, analysis_b)
        
        # Find unique requirements
        requirements_a = set(analysis_a.requirements_met + analysis_a.requirements_missing)
        requirements_b = set(analysis_b.requirements_met + analysis_b.requirements_missing)
        
        unique_a = list(requirements_a - requirements_b)
        unique_b = list(requirements_b - requirements_a)
        
        # Identify conflicting requirements (simplified)
        conflicting = self._identify_conflicting_requirements(analysis_a, analysis_b)
        
        # Determine jurisdiction advantage
        if compliance_diff > 0.1 and audit_readiness_diff > 0.1:
            advantage = analysis_b.authority_id
        elif compliance_diff < -0.1 and audit_readiness_diff < -0.1:
            advantage = analysis_a.authority_id
        else:
            advantage = "comparable"
        
        # Calculate complexity difference
        complexity_diff = self._calculate_complexity_difference(analysis_a, analysis_b)
        
        # Calculate cost-benefit ratio
        cost_benefit = self._calculate_comparison_cost_benefit(analysis_a, analysis_b)
        
        return AuthorityComparisonMetrics(
            authority_a=analysis_a.authority_id,
            authority_b=analysis_b.authority_id,
            compliance_score_diff=compliance_diff,
            audit_readiness_diff=audit_readiness_diff,
            penalty_risk_comparison=penalty_comparison,
            unique_requirements_a=unique_a,
            unique_requirements_b=unique_b,
            conflicting_requirements=conflicting,
            jurisdiction_advantage=advantage,
            complexity_difference=complexity_diff,
            cost_benefit_ratio=cost_benefit
        )
    
    def _identify_common_gaps(self, analyses: List[Big4ComplianceAnalysis]) -> List[str]:
        """Identify compliance gaps common across all authorities"""
        
        if not analyses:
            return []
        
        # Start with first analysis gaps
        common_gaps = set(analyses[0].requirements_missing)
        
        # Find intersection with all other analyses
        for analysis in analyses[1:]:
            common_gaps = common_gaps.intersection(set(analysis.requirements_missing))
        
        return list(common_gaps)
    
    def _identify_universal_requirements(self, analyses: List[Big4ComplianceAnalysis]) -> List[str]:
        """Identify requirements that all authorities consider important"""
        
        if not analyses:
            return []
        
        # Start with first analysis requirements
        universal_reqs = set(analyses[0].requirements_met + analyses[0].requirements_missing)
        
        # Find intersection with all other analyses
        for analysis in analyses[1:]:
            analysis_reqs = set(analysis.requirements_met + analysis.requirements_missing)
            universal_reqs = universal_reqs.intersection(analysis_reqs)
        
        return list(universal_reqs)
    
    def _identify_jurisdiction_specific_requirements(
        self, 
        analyses: List[Big4ComplianceAnalysis]
    ) -> Dict[str, List[str]]:
        """Identify requirements specific to each jurisdiction"""
        
        jurisdiction_requirements = {}
        
        for analysis in analyses:
            analysis_reqs = set(analysis.requirements_met + analysis.requirements_missing)
            
            # Find requirements unique to this authority
            unique_reqs = analysis_reqs.copy()
            for other_analysis in analyses:
                if other_analysis.authority_id != analysis.authority_id:
                    other_reqs = set(other_analysis.requirements_met + other_analysis.requirements_missing)
                    unique_reqs = unique_reqs - other_reqs
            
            jurisdiction_requirements[analysis.authority_id] = list(unique_reqs)
        
        return jurisdiction_requirements
    
    async def _generate_optimal_strategy(
        self,
        analyses: List[Big4ComplianceAnalysis],
        comparisons: List[AuthorityComparisonMetrics],
        industry: Optional[str],
        company_size: Optional[str]
    ) -> Dict[str, Any]:
        """Generate optimal compliance strategy across authorities"""
        
        # Find best performing authority
        best_authority = max(analyses, key=lambda a: a.compliance_score + a.audit_readiness_score)
        
        # Identify strategic insights
        strategy = {
            "recommended_primary_authority": best_authority.authority_id,
            "reasoning": [
                f"Highest compliance score: {best_authority.compliance_score:.2f}",
                f"Best audit readiness: {best_authority.audit_readiness_score:.2f}",
                f"Penalty risk level: {best_authority.penalty_risk_level}"
            ]
        }
        
        # Add industry-specific considerations
        if industry:
            strategy["industry_considerations"] = self._get_industry_strategic_insights(
                analyses, industry
            )
        
        # Add company size considerations
        if company_size:
            strategy["size_considerations"] = self._get_size_strategic_insights(
                analyses, company_size
            )
        
        # Multi-authority opportunities
        multi_auth_opportunities = self._identify_multi_authority_opportunities(comparisons)
        if multi_auth_opportunities:
            strategy["multi_authority_opportunities"] = multi_auth_opportunities
        
        return strategy
    
    def _create_implementation_roadmap(
        self,
        analyses: List[Big4ComplianceAnalysis],
        optimal_strategy: Dict[str, Any],
        common_gaps: List[str]
    ) -> List[Dict[str, Any]]:
        """Create implementation roadmap for optimal compliance strategy"""
        
        roadmap = []
        
        # Phase 1: Address common gaps (affects all jurisdictions)
        if common_gaps:
            roadmap.append({
                "phase": 1,
                "title": "Address Universal Compliance Gaps",
                "duration_weeks": 4,
                "priority": "high",
                "actions": [f"Implement {gap}" for gap in common_gaps[:5]],
                "impact": "Improves compliance across all jurisdictions",
                "cost_estimate": "Medium"
            })
        
        # Phase 2: Optimize for primary authority
        primary_authority_id = optimal_strategy["recommended_primary_authority"]
        primary_analysis = next(a for a in analyses if a.authority_id == primary_authority_id)
        
        roadmap.append({
            "phase": 2,
            "title": f"Optimize for {primary_analysis.authority_name}",
            "duration_weeks": 6,
            "priority": "high",
            "actions": primary_analysis.immediate_actions[:3],
            "impact": f"Maximizes compliance for primary jurisdiction",
            "cost_estimate": "High"
        })
        
        # Phase 3: Secondary optimizations
        roadmap.append({
            "phase": 3,
            "title": "Secondary Jurisdiction Optimizations",
            "duration_weeks": 8,
            "priority": "medium",
            "actions": [
                "Review cross-jurisdiction requirements",
                "Implement jurisdiction-specific measures",
                "Establish ongoing monitoring"
            ],
            "impact": "Ensures compliance across all relevant jurisdictions",
            "cost_estimate": "Medium"
        })
        
        return roadmap
    
    # Helper methods for calculations and comparisons
    def _compare_penalty_risk(
        self,
        analysis_a: Big4ComplianceAnalysis,
        analysis_b: Big4ComplianceAnalysis
    ) -> str:
        """Compare penalty risk levels between two authorities"""
        
        risk_levels = {"low": 1, "medium": 2, "high": 3, "critical": 4}
        
        risk_a = risk_levels.get(analysis_a.penalty_risk_level, 2)
        risk_b = risk_levels.get(analysis_b.penalty_risk_level, 2)
        
        if risk_a < risk_b:
            return f"{analysis_a.authority_id} has lower penalty risk"
        elif risk_a > risk_b:
            return f"{analysis_b.authority_id} has lower penalty risk"
        else:
            return "Comparable penalty risk levels"
    
    def _identify_conflicting_requirements(
        self,
        analysis_a: Big4ComplianceAnalysis,
        analysis_b: Big4ComplianceAnalysis
    ) -> List[Dict[str, Any]]:
        """Identify potentially conflicting requirements between authorities"""
        
        # Simplified conflict detection - can be enhanced
        conflicts = []
        
        # Check for requirements that one authority considers met and another considers missing
        a_met = set(analysis_a.requirements_met)
        b_missing = set(analysis_b.requirements_missing)
        
        potential_conflicts = a_met.intersection(b_missing)
        
        for conflict in potential_conflicts:
            conflicts.append({
                "requirement": conflict,
                "authority_a_status": "met",
                "authority_b_status": "missing",
                "severity": "medium"
            })
        
        return conflicts
    
    def _calculate_complexity_difference(
        self,
        analysis_a: Big4ComplianceAnalysis,
        analysis_b: Big4ComplianceAnalysis
    ) -> str:
        """Calculate implementation complexity difference"""
        
        # Simple heuristic based on number of missing requirements and immediate actions
        complexity_a = len(analysis_a.requirements_missing) + len(analysis_a.immediate_actions)
        complexity_b = len(analysis_b.requirements_missing) + len(analysis_b.immediate_actions)
        
        diff = abs(complexity_a - complexity_b)
        
        if diff < 2:
            return "comparable"
        elif complexity_a > complexity_b:
            return f"{analysis_a.authority_id} more complex"
        else:
            return f"{analysis_b.authority_id} more complex"
    
    def _calculate_comparison_cost_benefit(
        self,
        analysis_a: Big4ComplianceAnalysis,
        analysis_b: Big4ComplianceAnalysis
    ) -> float:
        """Calculate cost-benefit ratio for comparison"""
        
        # Simple heuristic - can be enhanced with real cost data
        benefit_a = analysis_a.compliance_score + analysis_a.audit_readiness_score
        benefit_b = analysis_b.compliance_score + analysis_b.audit_readiness_score
        
        cost_a = len(analysis_a.requirements_missing) * 0.1  # Arbitrary cost unit
        cost_b = len(analysis_b.requirements_missing) * 0.1
        
        ratio_a = benefit_a / max(cost_a, 0.1)
        ratio_b = benefit_b / max(cost_b, 0.1)
        
        return ratio_b / max(ratio_a, 0.1)  # Relative ratio
    
    # Configuration loading methods
    def _load_comparison_weights(self) -> Dict[str, float]:
        """Load weights for multi-authority comparison"""
        return {
            "compliance_score": 0.4,
            "audit_readiness": 0.3,
            "penalty_risk": 0.2,
            "implementation_complexity": 0.1
        }
    
    def _load_jurisdiction_cost_data(self) -> Dict[str, Dict[str, Any]]:
        """Load jurisdiction-specific cost data"""
        return {
            "bfdi": {
                "base_compliance_cost": 50000,
                "complexity_multiplier": 1.2,
                "typical_timeline_weeks": 12
            },
            "baylda": {
                "base_compliance_cost": 35000,
                "complexity_multiplier": 1.0,
                "typical_timeline_weeks": 8
            },
            "lfd_bw": {
                "base_compliance_cost": 40000,
                "complexity_multiplier": 1.1,
                "typical_timeline_weeks": 10
            },
            "ldi_nrw": {
                "base_compliance_cost": 38000,
                "complexity_multiplier": 1.05,
                "typical_timeline_weeks": 9
            }
        }
    
    def _calculate_compliance_costs(
        self,
        analyses: List[Big4ComplianceAnalysis],
        company_size: Optional[str]
    ) -> Dict[str, str]:
        """Calculate estimated compliance costs for each authority"""
        
        cost_estimates = {}
        size_multiplier = {"small": 0.7, "medium": 1.0, "large": 1.5}.get(company_size, 1.0)
        
        for analysis in analyses:
            jurisdiction_data = self.jurisdiction_costs.get(analysis.authority_id, {})
            base_cost = jurisdiction_data.get("base_compliance_cost", 40000)
            complexity_mult = jurisdiction_data.get("complexity_multiplier", 1.0)
            
            # Adjust for missing requirements
            missing_count = len(analysis.requirements_missing)
            missing_cost = missing_count * 5000  # €5k per missing requirement
            
            # Adjust for priority gaps
            priority_gap_cost = len(analysis.priority_gaps) * 8000  # €8k per priority gap
            
            total_cost = (base_cost + missing_cost + priority_gap_cost) * complexity_mult * size_multiplier
            
            # Format as range
            lower_bound = int(total_cost * 0.8)
            upper_bound = int(total_cost * 1.2)
            
            cost_estimates[analysis.authority_id] = f"€{lower_bound:,} - €{upper_bound:,}"
        
        return cost_estimates
    
    def _calculate_risk_reduction_potential(
        self,
        analyses: List[Big4ComplianceAnalysis]
    ) -> Dict[str, float]:
        """Calculate risk reduction potential for each authority"""
        
        risk_reduction = {}
        
        for analysis in analyses:
            # Base risk reduction from compliance score
            base_reduction = analysis.compliance_score * 0.8
            
            # Additional reduction from audit readiness
            audit_reduction = analysis.audit_readiness_score * 0.2
            
            total_reduction = min(1.0, base_reduction + audit_reduction)
            risk_reduction[analysis.authority_id] = total_reduction
        
        return risk_reduction
    
    def _calculate_roi_analysis(
        self,
        cost_estimates: Dict[str, str],
        risk_reduction: Dict[str, float]
    ) -> Dict[str, Any]:
        """Calculate ROI analysis for compliance investments"""
        
        roi_analysis = {}
        
        for authority_id in cost_estimates:
            # Extract average cost (simplified)
            cost_str = cost_estimates[authority_id]
            # Parse cost range and take average
            costs = [int(x.replace('€', '').replace(',', '')) for x in cost_str.split(' - ')]
            avg_cost = sum(costs) / len(costs)
            
            # Calculate potential penalty avoidance (estimated)
            risk_reduction_pct = risk_reduction.get(authority_id, 0.5)
            potential_penalty = 200000  # Average penalty estimate
            penalty_avoidance = potential_penalty * risk_reduction_pct
            
            # Calculate simple ROI
            roi = (penalty_avoidance - avg_cost) / avg_cost if avg_cost > 0 else 0
            
            roi_analysis[authority_id] = {
                "investment_cost": avg_cost,
                "potential_penalty_avoidance": penalty_avoidance,
                "roi_percentage": roi * 100,
                "payback_period_months": 12 if roi > 0 else 36,
                "risk_reduction_percentage": risk_reduction_pct * 100
            }
        
        return roi_analysis
    
    def _determine_primary_recommendation(
        self,
        analyses: List[Big4ComplianceAnalysis],
        comparisons: List[AuthorityComparisonMetrics],
        optimal_strategy: Dict[str, Any]
    ) -> str:
        """Determine primary authority recommendation"""
        
        # Use optimal strategy recommendation
        return optimal_strategy.get("recommended_primary_authority", analyses[0].authority_id)
    
    def _evaluate_multi_authority_strategy(
        self,
        analyses: List[Big4ComplianceAnalysis],
        comparisons: List[AuthorityComparisonMetrics],
        cost_estimates: Dict[str, str]
    ) -> Optional[str]:
        """Evaluate if multi-authority strategy is beneficial"""
        
        # Check if there are significant advantages to multiple authorities
        score_variance = self._calculate_score_variance(analyses)
        
        if score_variance > 0.2:  # Significant variance in compliance scores
            return "Consider multi-authority approach to leverage specific strengths"
        
        # Check for complementary strengths
        complementary_strengths = self._identify_complementary_strengths(analyses)
        if complementary_strengths:
            return f"Multi-authority strategy recommended: {complementary_strengths}"
        
        return None  # Single authority approach recommended
    
    def _calculate_score_variance(self, analyses: List[Big4ComplianceAnalysis]) -> float:
        """Calculate variance in compliance scores"""
        scores = [analysis.compliance_score for analysis in analyses]
        avg_score = sum(scores) / len(scores)
        variance = sum((score - avg_score) ** 2 for score in scores) / len(scores)
        return variance ** 0.5  # Standard deviation
    
    def _identify_complementary_strengths(self, analyses: List[Big4ComplianceAnalysis]) -> Optional[str]:
        """Identify complementary strengths across authorities"""
        
        # Look for authorities with different strong areas
        strong_areas = {}
        
        for analysis in analyses:
            if analysis.compliance_score > 0.8:
                strong_areas[analysis.authority_id] = "high_compliance"
            elif analysis.audit_readiness_score > 0.8:
                strong_areas[analysis.authority_id] = "audit_ready"
            elif analysis.penalty_risk_level == "low":
                strong_areas[analysis.authority_id] = "low_risk"
        
        if len(strong_areas) >= 2:
            strengths = ", ".join([f"{k}: {v}" for k, v in strong_areas.items()])
            return f"Complementary strengths identified: {strengths}"
        
        return None
    
    def _get_industry_strategic_insights(
        self,
        analyses: List[Big4ComplianceAnalysis],
        industry: str
    ) -> List[str]:
        """Get industry-specific strategic insights"""
        
        insights = []
        
        industry_authority_map = {
            "automotive": ["baylda", "lfd_bw"],
            "software": ["lfd_bw", "bfdi"],
            "manufacturing": ["ldi_nrw", "baylda"],
            "healthcare": ["bfdi", "lfd_bw"]
        }
        
        preferred_authorities = industry_authority_map.get(industry, [])
        
        for analysis in analyses:
            if analysis.authority_id in preferred_authorities:
                insights.append(f"{analysis.authority_name} has specific expertise in {industry}")
        
        return insights
    
    def _get_size_strategic_insights(
        self,
        analyses: List[Big4ComplianceAnalysis],
        company_size: str
    ) -> List[str]:
        """Get company size-specific strategic insights"""
        
        insights = []
        
        # SME-focused authorities
        sme_focused = ["baylda", "lfd_bw", "ldi_nrw"]
        
        if company_size in ["small", "medium"]:
            for analysis in analyses:
                if analysis.authority_id in sme_focused:
                    insights.append(f"{analysis.authority_name} provides SME-focused guidance")
        elif company_size == "large":
            for analysis in analyses:
                if analysis.authority_id == "bfdi":
                    insights.append(f"{analysis.authority_name} specializes in large enterprise compliance")
        
        return insights
    
    def _identify_multi_authority_opportunities(
        self,
        comparisons: List[AuthorityComparisonMetrics]
    ) -> Optional[List[str]]:
        """Identify opportunities for multi-authority strategies"""
        
        opportunities = []
        
        for comparison in comparisons:
            if comparison.jurisdiction_advantage == "comparable":
                opportunities.append(
                    f"Consider leveraging both {comparison.authority_a} and {comparison.authority_b} "
                    f"for their respective strengths"
                )
        
        return opportunities if opportunities else None
    
    def _estimate_switching_effort(
        self,
        current_analysis: Big4ComplianceAnalysis,
        target_analysis: Big4ComplianceAnalysis
    ) -> str:
        """Estimate effort required to switch jurisdictions"""
        
        # Calculate differences in requirements
        current_reqs = set(current_analysis.requirements_met + current_analysis.requirements_missing)
        target_reqs = set(target_analysis.requirements_met + target_analysis.requirements_missing)
        
        new_requirements = len(target_reqs - current_reqs)
        
        if new_requirements < 3:
            return "low"
        elif new_requirements < 7:
            return "medium"
        else:
            return "high"
    
    def _calculate_switching_cost_benefit(
        self,
        current_analysis: Big4ComplianceAnalysis,
        target_analysis: Big4ComplianceAnalysis
    ) -> float:
        """Calculate cost-benefit ratio for switching jurisdictions"""
        
        # Benefit: improvement in compliance and audit readiness
        compliance_improvement = target_analysis.compliance_score - current_analysis.compliance_score
        audit_improvement = target_analysis.audit_readiness_score - current_analysis.audit_readiness_score
        total_benefit = compliance_improvement + audit_improvement
        
        # Cost: estimated switching effort (simplified)
        switching_effort = self._estimate_switching_effort(current_analysis, target_analysis)
        effort_cost = {"low": 0.1, "medium": 0.3, "high": 0.6}.get(switching_effort, 0.3)
        
        # Return benefit/cost ratio
        return total_benefit / max(effort_cost, 0.1)