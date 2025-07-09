# app/services/german_authority_engine/__init__.py
"""
German Authority Mapping Engine

Comprehensive database and analysis engine for German data protection authorities.
Covers all 16 German states plus federal level with industry-specific requirements.

Key Components:
- Authority Profiles: Complete data for all German DPAs
- Authority Analyzer: Core compliance analysis engine
- Requirement Mapper: Map content to authority requirements
- Compliance Scorer: Calculate authority-specific scores
- Authority Detector: Auto-detect relevant authority

Usage:
    from app.services.german_authority_engine import GermanAuthorityEngine
    
    engine = GermanAuthorityEngine()
    analysis = await engine.analyze_for_authority(documents, "baylda", "automotive")
"""

from .authority_analyzer import GermanAuthorityEngine, AuthorityComplianceAnalysis
from .authority_profiles import (
    GermanAuthority, 
    AuthorityProfile, 
    AuthorityRequirement,
    get_authority_profile,
    get_all_authorities
)
from .authority_detector import AuthorityDetector
from .requirement_mapper import RequirementMapper
from .compliance_scorer import ComplianceScorer

__all__ = [
    # Main engine
    "GermanAuthorityEngine",
    "AuthorityComplianceAnalysis",
    
    # Authority data
    "GermanAuthority",
    "AuthorityProfile", 
    "AuthorityRequirement",
    "get_authority_profile",
    "get_all_authorities",
    
    # Core components
    "AuthorityDetector",
    "RequirementMapper", 
    "ComplianceScorer"
]

__version__ = "1.0.0"
__author__ = "WolfMerge Team"
__description__ = "German Data Protection Authority Intelligence Engine"