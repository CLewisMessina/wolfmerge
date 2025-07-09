# app/services/german_authority_engine/authority_profiles.py
"""
German Data Protection Authority Profiles

Comprehensive database of all German data protection authorities with
detailed compliance requirements, enforcement patterns, and contact information.

Covers:
- Federal level (BfDI)
- All 16 German states
- Industry-specific requirements
- Enforcement patterns and penalty ranges
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Union
from enum import Enum

class GermanAuthority(str, Enum):
    """All German data protection authorities"""
    # Federal
    BFDI = "bfdi"
    
    # States (alphabetical by German name)
    BADEN_WURTTEMBERG = "lfd_bw"
    BAYERN = "baylda" 
    BERLIN = "bfdi_berlin"
    BRANDENBURG = "lda_brandenburg"
    BREMEN = "lfd_bremen"
    HAMBURG = "bfdi_hamburg"
    HESSEN = "hbdi_hessen"
    MECKLENBURG_VORPOMMERN = "lfd_mv"
    NIEDERSACHSEN = "lfd_niedersachsen"
    NORDRHEIN_WESTFALEN = "ldi_nrw"
    RHEINLAND_PFALZ = "lfd_rlp"
    SAARLAND = "uld_saarland"
    SACHSEN = "saechsdsb"
    SACHSEN_ANHALT = "lfd_sachsen_anhalt"
    SCHLESWIG_HOLSTEIN = "uld_sh"
    THURINGEN = "tlfd_th"

@dataclass
class AuthorityRequirement:
    """Specific requirement from a German data protection authority"""
    article_reference: str
    requirement_text: str
    enforcement_priority: str  # high, medium, low
    typical_audit_focus: List[str]
    penalty_range: str
    industry_specific: List[str]
    german_legal_reference: Optional[str] = None
    implementation_guidance: Optional[str] = None
    
    def applies_to_industry(self, industry: str) -> bool:
        """Check if requirement applies to specific industry"""
        return not self.industry_specific or industry in self.industry_specific

@dataclass 
class AuthorityProfile:
    """Complete profile for German data protection authority"""
    authority_id: str
    name: str
    name_english: str
    jurisdiction: str
    state_code: str  # Two-letter state code
    
    # Contact information
    address: str
    phone: str
    email: str
    website: str
    
    # Authority characteristics
    specific_requirements: List[AuthorityRequirement]
    audit_patterns: Dict[str, str]
    industry_focus: List[str]
    enforcement_style: str
    penalty_approach: str
    
    # Regional specifics
    cross_border_coordination: List[str]
    recent_enforcement_examples: List[str]
    guidance_documents: List[str]
    
    # Business environment
    major_cities: List[str]
    key_industries: List[str]
    economic_profile: str

class AuthorityDatabase:
    """Centralized database for all German authority profiles"""
    
    def __init__(self):
        self._profiles: Dict[GermanAuthority, AuthorityProfile] = {}
        self._load_authority_profiles()
    
    def _load_authority_profiles(self):
        """Load all authority profiles from data modules"""
        # Import authority data modules
        from app.data.german_authorities.federal_authorities import get_federal_authorities
        from app.data.german_authorities.state_authorities_south import get_southern_authorities
        from app.data.german_authorities.state_authorities_north import get_northern_authorities
        from app.data.german_authorities.state_authorities_east import get_eastern_authorities
        from app.data.german_authorities.state_authorities_west import get_western_authorities
        
        # Load federal authorities
        federal_profiles = get_federal_authorities()
        self._profiles.update(federal_profiles)
        
        # Load state authorities by region
        southern_profiles = get_southern_authorities()
        self._profiles.update(southern_profiles)
        
        northern_profiles = get_northern_authorities()
        self._profiles.update(northern_profiles)
        
        eastern_profiles = get_eastern_authorities()
        self._profiles.update(eastern_profiles)
        
        western_profiles = get_western_authorities()
        self._profiles.update(western_profiles)
    
    def get_profile(self, authority: Union[str, GermanAuthority]) -> Optional[AuthorityProfile]:
        """Get authority profile by ID"""
        if isinstance(authority, str):
            try:
                authority = GermanAuthority(authority.lower())
            except ValueError:
                return None
        
        return self._profiles.get(authority)
    
    def get_all_profiles(self) -> Dict[GermanAuthority, AuthorityProfile]:
        """Get all authority profiles"""
        return self._profiles.copy()
    
    def get_authorities_by_region(self, region: str) -> List[AuthorityProfile]:
        """Get authorities by region (north, south, east, west)"""
        region_mapping = {
            "north": ["schleswig_holstein", "hamburg", "bremen", "niedersachsen", "mecklenburg_vorpommern"],
            "south": ["baden_wurttemberg", "bayern", "rheinland_pfalz", "saarland"],
            "east": ["berlin", "brandenburg", "sachsen", "sachsen_anhalt", "thuringen"],
            "west": ["nordrhein_westfalen", "hessen"]
        }
        
        state_codes = region_mapping.get(region.lower(), [])
        return [
            profile for profile in self._profiles.values()
            if profile.state_code.lower() in state_codes
        ]
    
    def get_authorities_by_industry(self, industry: str) -> List[AuthorityProfile]:
        """Get authorities with specific industry focus"""
        return [
            profile for profile in self._profiles.values()
            if industry.lower() in [ind.lower() for ind in profile.industry_focus]
        ]
    
    def search_authorities(
        self, 
        industry: Optional[str] = None,
        region: Optional[str] = None,
        enforcement_style: Optional[str] = None
    ) -> List[AuthorityProfile]:
        """Search authorities by multiple criteria"""
        results = list(self._profiles.values())
        
        if industry:
            results = [
                profile for profile in results
                if industry.lower() in [ind.lower() for ind in profile.industry_focus]
            ]
        
        if region:
            region_authorities = self.get_authorities_by_region(region)
            results = [
                profile for profile in results
                if profile in region_authorities
            ]
        
        if enforcement_style:
            results = [
                profile for profile in results
                if enforcement_style.lower() in profile.enforcement_style.lower()
            ]
        
        return results

# Global authority database instance
_authority_database = None

def get_authority_database() -> AuthorityDatabase:
    """Get global authority database instance"""
    global _authority_database
    if _authority_database is None:
        _authority_database = AuthorityDatabase()
    return _authority_database

def get_authority_profile(authority: Union[str, GermanAuthority]) -> Optional[AuthorityProfile]:
    """Convenience function to get authority profile"""
    return get_authority_database().get_profile(authority)

def get_all_authorities() -> Dict[GermanAuthority, AuthorityProfile]:
    """Convenience function to get all authorities"""
    return get_authority_database().get_all_profiles()

def get_authorities_by_industry(industry: str) -> List[AuthorityProfile]:
    """Convenience function to get authorities by industry"""
    return get_authority_database().get_authorities_by_industry(industry)

def get_authorities_by_region(region: str) -> List[AuthorityProfile]:
    """Convenience function to get authorities by region"""
    return get_authority_database().get_authorities_by_region(region)