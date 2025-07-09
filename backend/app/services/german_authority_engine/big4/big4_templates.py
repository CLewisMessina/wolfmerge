# backend/app/services/german_authority_engine/big4/big4_templates.py
"""
Big 4 Industry-Specific Compliance Templates

Pre-configured compliance templates for major German industries:
- Automotive (Bavaria/Baden-Württemberg focus)
- Software (Baden-Württemberg focus)
- Manufacturing (NRW focus)
- Healthcare (Multi-authority)

Each template includes authority-specific requirements, common compliance
patterns, and industry best practices.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from enum import Enum

from .big4_profiles import Big4Authority

class IndustryTemplate(str, Enum):
    """Supported industry templates"""
    AUTOMOTIVE = "automotive"
    SOFTWARE = "software"
    MANUFACTURING = "manufacturing"
    HEALTHCARE = "healthcare"
    FINTECH = "fintech"
    LOGISTICS = "logistics"

@dataclass
class ComplianceTemplate:
    """Industry-specific compliance template"""
    template_id: str
    industry: IndustryTemplate
    name: str
    description: str
    
    # Authority focus
    primary_authorities: List[Big4Authority]
    authority_priorities: Dict[str, float]
    
    # Required documents
    required_documents: List[Dict[str, Any]]
    recommended_documents: List[Dict[str, Any]]
    
    # Compliance areas
    critical_compliance_areas: List[Dict[str, Any]]
    industry_specific_requirements: List[Dict[str, Any]]
    
    # Common patterns
    typical_violations: List[str]
    best_practices: List[str]
    
    # Risk factors
    high_risk_activities: List[str]
    compliance_complexity: str  # low, medium, high, very_high

class Big4IndustryTemplateEngine:
    """
    Industry template engine for Big 4 authority compliance
    
    Provides pre-configured templates with industry-specific requirements,
    authority priorities, and compliance guidance.
    """
    
    def __init__(self):
        self.templates = self._load_industry_templates()
        self.authority_templates = self._load_authority_specific_templates()
    
    def get_template(self, industry: IndustryTemplate) -> Optional[ComplianceTemplate]:
        """Get compliance template for specific industry"""
        return self.templates.get(industry)
    
    def get_authority_template(
        self, 
        industry: IndustryTemplate, 
        authority: Big4Authority
    ) -> Optional[Dict[str, Any]]:
        """Get authority-specific template for industry"""
        template_key = f"{industry.value}_{authority.value}"
        return self.authority_templates.get(template_key)
    
    def get_recommended_authorities(self, industry: IndustryTemplate) -> List[Big4Authority]:
        """Get recommended authorities for industry (ordered by relevance)"""
        template = self.get_template(industry)
        if template:
            return template.primary_authorities
        return [Big4Authority.BFDI]  # Default fallback
    
    def get_industry_requirements_checklist(
        self, 
        industry: IndustryTemplate,
        authority: Big4Authority
    ) -> Dict[str, Any]:
        """Generate industry and authority-specific requirements checklist"""
        
        template = self.get_template(industry)
        authority_template = self.get_authority_template(industry, authority)
        
        if not template:
            return self._get_generic_checklist(authority)
        
        checklist = {
            "industry": industry.value,
            "authority": authority.value,
            "template_name": template.name,
            "compliance_complexity": template.compliance_complexity,
            
            "required_documents": template.required_documents,
            "critical_areas": template.critical_compliance_areas,
            "industry_requirements": template.industry_specific_requirements,
            
            "authority_specific": authority_template or {},
            
            "risk_assessment": {
                "high_risk_activities": template.high_risk_activities,
                "typical_violations": template.typical_violations,
                "complexity_rating": template.compliance_complexity
            },
            
            "guidance": {
                "best_practices": template.best_practices,
                "authority_focus": self._get_authority_focus_areas(authority, template),
                "implementation_priority": self._get_implementation_priorities(template, authority)
            }
        }
        
        return checklist
    
    def _load_industry_templates(self) -> Dict[IndustryTemplate, ComplianceTemplate]:
        """Load all industry compliance templates"""
        
        templates = {}
        
        # Automotive Template (Bavaria/Baden-Württemberg focus)
        templates[IndustryTemplate.AUTOMOTIVE] = ComplianceTemplate(
            template_id="automotive_de",
            industry=IndustryTemplate.AUTOMOTIVE,
            name="German Automotive GDPR Compliance",
            description="Compliance template for automotive companies in Germany, focusing on connected vehicles and supply chain data protection",
            
            primary_authorities=[Big4Authority.BAYLDA, Big4Authority.LFD_BW, Big4Authority.LDI_NRW],
            authority_priorities={
                "baylda": 1.0,      # Primary for Bavaria automotive
                "lfd_bw": 0.9,      # Strong for Baden-Württemberg
                "ldi_nrw": 0.7,     # Manufacturing support
                "bfdi": 0.5         # Federal oversight
            },
            
            required_documents=[
                {
                    "document_type": "privacy_policy_vehicles",
                    "name": "Connected Vehicle Privacy Policy",
                    "description": "Privacy policy covering vehicle data collection, processing, and sharing",
                    "authority_requirements": ["baylda", "lfd_bw"],
                    "compliance_level": "mandatory"
                },
                {
                    "document_type": "dpa_suppliers",
                    "name": "Automotive Supplier Data Processing Agreements",
                    "description": "DPAs with all automotive suppliers handling personal data",
                    "authority_requirements": ["baylda", "lfd_bw", "ldi_nrw"],
                    "compliance_level": "mandatory"
                },
                {
                    "document_type": "consent_management",
                    "name": "Vehicle Data Consent Management System",
                    "description": "Technical implementation for managing driver/passenger consent",
                    "authority_requirements": ["baylda"],
                    "compliance_level": "mandatory"
                }
            ],
            
            recommended_documents=[
                {
                    "document_type": "dpia_connected_vehicles",
                    "name": "Connected Vehicle DPIA",
                    "description": "Data protection impact assessment for connected vehicle systems",
                    "compliance_level": "recommended"
                },
                {
                    "document_type": "technical_measures_doc",
                    "name": "Technical Data Protection Measures Documentation", 
                    "description": "Documentation of technical measures for vehicle data protection",
                    "compliance_level": "recommended"
                }
            ],
            
            critical_compliance_areas=[
                {
                    "area": "vehicle_data_consent",
                    "description": "Consent management for vehicle data collection",
                    "gdpr_articles": ["Art. 6", "Art. 7"],
                    "authority_focus": ["baylda"],
                    "risk_level": "high",
                    "common_violations": ["Invalid consent mechanisms", "Unclear consent scope"]
                },
                {
                    "area": "location_tracking",
                    "description": "Location data processing and retention",
                    "gdpr_articles": ["Art. 6", "Art. 9", "Art. 17"],
                    "authority_focus": ["baylda", "lfd_bw"],
                    "risk_level": "very_high",
                    "common_violations": ["Excessive location data retention", "No opt-out mechanisms"]
                },
                {
                    "area": "supplier_data_sharing",
                    "description": "Data sharing with automotive suppliers and partners",
                    "gdpr_articles": ["Art. 28", "Art. 44-49"],
                    "authority_focus": ["baylda", "lfd_bw", "ldi_nrw"],
                    "risk_level": "high",
                    "common_violations": ["Missing DPAs", "Inadequate supplier oversight"]
                }
            ],
            
            industry_specific_requirements=[
                {
                    "requirement": "connected_vehicle_transparency",
                    "description": "Clear information about data collection in connected vehicles",
                    "implementation": "In-vehicle privacy notices and mobile app disclosures"
                },
                {
                    "requirement": "automotive_data_minimization",
                    "description": "Data minimization specific to automotive use cases",
                    "implementation": "Limit data collection to essential vehicle functions"
                },
                {
                    "requirement": "telematics_consent",
                    "description": "Specific consent requirements for telematics data",
                    "implementation": "Granular consent for different telematics services"
                }
            ],
            
            typical_violations=[
                "Collecting vehicle data without proper consent",
                "Sharing location data with third parties without legal basis",
                "Inadequate supplier data processing agreements",
                "Missing privacy notices in vehicle interfaces",
                "Excessive retention of diagnostic data"
            ],
            
            best_practices=[
                "Implement privacy by design in vehicle systems",
                "Use pseudonymization for vehicle identifier data",
                "Provide granular consent options for different vehicle services",
                "Regular supplier compliance audits and assessments",
                "Clear data retention policies for different data types"
            ],
            
            high_risk_activities=[
                "Cross-border vehicle data transfers",
                "Third-party data sharing for marketing",
                "Biometric data collection (driver monitoring)",
                "Location tracking for non-essential services",
                "Automated decision-making based on driving behavior"
            ],
            
            compliance_complexity="very_high"
        )
        
        # Software Template (Baden-Württemberg focus)
        templates[IndustryTemplate.SOFTWARE] = ComplianceTemplate(
            template_id="software_de",
            industry=IndustryTemplate.SOFTWARE,
            name="German Software Industry GDPR Compliance",
            description="Compliance template for software companies, focusing on privacy by design and technical implementation",
            
            primary_authorities=[Big4Authority.LFD_BW, Big4Authority.BFDI, Big4Authority.BAYLDA],
            authority_priorities={
                "lfd_bw": 1.0,      # Primary for software companies
                "bfdi": 0.8,        # Federal tech oversight
                "baylda": 0.6,      # If serving automotive
                "ldi_nrw": 0.4      # Manufacturing software
            },
            
            required_documents=[
                {
                    "document_type": "privacy_by_design_doc",
                    "name": "Privacy by Design Implementation Documentation",
                    "description": "Documentation of privacy by design in software development lifecycle",
                    "authority_requirements": ["lfd_bw"],
                    "compliance_level": "mandatory"
                },
                {
                    "document_type": "api_privacy_policy",
                    "name": "API Data Protection Policy",
                    "description": "Privacy policy covering API data collection and processing",
                    "authority_requirements": ["lfd_bw", "bfdi"],
                    "compliance_level": "mandatory"
                },
                {
                    "document_type": "user_consent_system",
                    "name": "User Consent Management System",
                    "description": "Technical implementation of consent management",
                    "authority_requirements": ["lfd_bw"],
                    "compliance_level": "mandatory"
                }
            ],
            
            recommended_documents=[
                {
                    "document_type": "code_review_privacy",
                    "name": "Privacy-Focused Code Review Guidelines",
                    "description": "Guidelines for privacy considerations in code reviews",
                    "compliance_level": "recommended"
                }
            ],
            
            critical_compliance_areas=[
                {
                    "area": "privacy_by_design",
                    "description": "Privacy by design in software development",
                    "gdpr_articles": ["Art. 25"],
                    "authority_focus": ["lfd_bw"],
                    "risk_level": "high",
                    "common_violations": ["Lack of privacy by design documentation", "Retroactive privacy measures"]
                },
                {
                    "area": "user_consent_technical",
                    "description": "Technical implementation of user consent",
                    "gdpr_articles": ["Art. 7"],
                    "authority_focus": ["lfd_bw"],
                    "risk_level": "high",
                    "common_violations": ["Dark patterns", "Unclear consent interfaces"]
                }
            ],
            
            industry_specific_requirements=[
                {
                    "requirement": "development_lifecycle_privacy",
                    "description": "Privacy integration throughout software development lifecycle",
                    "implementation": "Privacy requirements in each development phase"
                },
                {
                    "requirement": "api_data_protection",
                    "description": "Data protection measures for API endpoints",
                    "implementation": "Authentication, authorization, and data minimization in APIs"
                },
                {
                    "requirement": "automated_privacy_compliance",
                    "description": "Automated compliance checking in CI/CD pipelines",
                    "implementation": "Privacy compliance tests in automated testing"
                }
            ],
            
            typical_violations=[
                "Lack of privacy by design documentation",
                "Inadequate consent management implementation",
                "Missing data protection in API design",
                "Insufficient user data access controls",
                "Poor privacy policy implementation in software"
            ],
            
            best_practices=[
                "Integrate privacy requirements in agile development",
                "Implement privacy-first API design",
                "Use privacy-preserving analytics",
                "Regular privacy impact assessments for new features",
                "Automated privacy compliance testing"
            ],
            
            high_risk_activities=[
                "Cross-border data transfers through APIs",
                "Automated decision-making algorithms",
                "User behavior tracking and profiling",
                "Third-party integrations and data sharing",
                "AI/ML model training on personal data"
            ],
            
            compliance_complexity="high"
        )
        
        # Manufacturing Template (NRW focus)
        templates[IndustryTemplate.MANUFACTURING] = ComplianceTemplate(
            template_id="manufacturing_de",
            industry=IndustryTemplate.MANUFACTURING,
            name="German Manufacturing GDPR Compliance",
            description="Compliance template for manufacturing companies, focusing on industrial IoT and operational data",
            
            primary_authorities=[Big4Authority.LDI_NRW, Big4Authority.BAYLDA, Big4Authority.LFD_BW],
            authority_priorities={
                "ldi_nrw": 1.0,     # Primary for manufacturing
                "baylda": 0.7,      # Manufacturing in Bavaria
                "lfd_bw": 0.6,      # Manufacturing in BW
                "bfdi": 0.5         # Federal oversight
            },
            
            required_documents=[
                {
                    "document_type": "manufacturing_privacy_policy",
                    "name": "Manufacturing Operations Privacy Policy",
                    "description": "Privacy policy covering employee and operational data in manufacturing",
                    "authority_requirements": ["ldi_nrw"],
                    "compliance_level": "mandatory"
                },
                {
                    "document_type": "iot_data_processing",
                    "name": "Industrial IoT Data Processing Documentation",
                    "description": "Documentation of IoT device data collection and processing",
                    "authority_requirements": ["ldi_nrw", "baylda"],
                    "compliance_level": "mandatory"
                },
                {
                    "document_type": "employee_monitoring_policy",
                    "name": "Employee Monitoring and Tracking Policy",
                    "description": "Policy covering employee monitoring in manufacturing environments",
                    "authority_requirements": ["ldi_nrw"],
                    "compliance_level": "mandatory"
                }
            ],
            
            recommended_documents=[
                {
                    "document_type": "supply_chain_privacy",
                    "name": "Supply Chain Privacy Requirements",
                    "description": "Privacy requirements for supply chain partners",
                    "compliance_level": "recommended"
                }
            ],
            
            critical_compliance_areas=[
                {
                    "area": "employee_monitoring",
                    "description": "Employee monitoring and workplace surveillance",
                    "gdpr_articles": ["Art. 6", "Art. 88"],
                    "authority_focus": ["ldi_nrw"],
                    "risk_level": "high",
                    "common_violations": ["Excessive employee monitoring", "Lack of transparency"]
                },
                {
                    "area": "industrial_iot",
                    "description": "Industrial IoT device data collection",
                    "gdpr_articles": ["Art. 5", "Art. 25"],
                    "authority_focus": ["ldi_nrw", "baylda"],
                    "risk_level": "medium",
                    "common_violations": ["Unsecured IoT devices", "Excessive data collection"]
                }
            ],
            
            industry_specific_requirements=[
                {
                    "requirement": "manufacturing_risk_assessment",
                    "description": "Risk assessment for manufacturing data processing",
                    "implementation": "Regular assessment of manufacturing process privacy risks"
                },
                {
                    "requirement": "operational_data_minimization",
                    "description": "Data minimization in manufacturing operations",
                    "implementation": "Limit operational data collection to business necessity"
                }
            ],
            
            typical_violations=[
                "Excessive employee monitoring without legal basis",
                "Insecure industrial IoT device configurations",
                "Missing privacy notices for operational data collection",
                "Inadequate access controls for manufacturing systems",
                "Poor data retention policies for operational data"
            ],
            
            best_practices=[
                "Implement privacy by design in manufacturing systems",
                "Regular IoT security and privacy assessments",
                "Clear employee privacy policies and training",
                "Secure manufacturing data storage and processing",
                "Regular supplier privacy compliance checks"
            ],
            
            high_risk_activities=[
                "Biometric employee authentication systems",
                "Video surveillance in manufacturing areas",
                "Cross-border manufacturing data transfers",
                "Third-party maintenance access to systems",
                "Predictive analytics on employee data"
            ],
            
            compliance_complexity="high"
        )
        
        # Healthcare Template (Multi-authority)
        templates[IndustryTemplate.HEALTHCARE] = ComplianceTemplate(
            template_id="healthcare_de",
            industry=IndustryTemplate.HEALTHCARE,
            name="German Healthcare GDPR Compliance",
            description="Compliance template for healthcare organizations, covering patient data and medical research",
            
            primary_authorities=[Big4Authority.BFDI, Big4Authority.LFD_BW, Big4Authority.BAYLDA],
            authority_priorities={
                "bfdi": 1.0,        # Federal health oversight
                "lfd_bw": 0.8,      # Health tech companies
                "baylda": 0.7,      # Healthcare in Bavaria
                "ldi_nrw": 0.6      # Healthcare in NRW
            },
            
            required_documents=[
                {
                    "document_type": "patient_privacy_policy",
                    "name": "Patient Data Privacy Policy",
                    "description": "Comprehensive privacy policy for patient data processing",
                    "authority_requirements": ["bfdi", "lfd_bw", "baylda"],
                    "compliance_level": "mandatory"
                },
                {
                    "document_type": "medical_consent_forms",
                    "name": "Medical Data Consent Forms",
                    "description": "Specific consent forms for medical data processing",
                    "authority_requirements": ["bfdi"],
                    "compliance_level": "mandatory"
                }
            ],
            
            recommended_documents=[],
            
            critical_compliance_areas=[
                {
                    "area": "patient_consent",
                    "description": "Patient consent for medical data processing",
                    "gdpr_articles": ["Art. 7", "Art. 9"],
                    "authority_focus": ["bfdi"],
                    "risk_level": "very_high",
                    "common_violations": ["Invalid consent for special category data"]
                }
            ],
            
            industry_specific_requirements=[
                {
                    "requirement": "medical_data_special_categories",
                    "description": "Special protection for medical data (Art. 9)",
                    "implementation": "Enhanced consent and security for health data"
                }
            ],
            
            typical_violations=[
                "Processing health data without adequate legal basis",
                "Inadequate consent for medical research",
                "Insecure patient data storage and transmission"
            ],
            
            best_practices=[
                "Implement enhanced security for health data",
                "Clear patient consent processes",
                "Regular staff privacy training"
            ],
            
            high_risk_activities=[
                "Medical research data processing",
                "Cross-border patient data transfers",
                "AI-based medical diagnosis systems"
            ],
            
            compliance_complexity="very_high"
        )
        
        return templates
    
    def _load_authority_specific_templates(self) -> Dict[str, Dict[str, Any]]:
        """Load authority-specific template enhancements"""
        
        return {
            "automotive_baylda": {
                "enforcement_focus": [
                    "Connected vehicle consent mechanisms",
                    "Automotive supplier compliance",
                    "Location data processing"
                ],
                "specific_requirements": [
                    "Bavarian automotive data protection guidelines compliance",
                    "Telematics consent management system",
                    "Automotive supply chain DPA standardization"
                ],
                "penalty_considerations": [
                    "BayLDA focuses on technical implementation quality",
                    "Higher penalties for consent mechanism violations",
                    "Automotive industry receives increased scrutiny"
                ]
            },
            
            "automotive_lfd_bw": {
                "enforcement_focus": [
                    "Privacy by design in automotive systems",
                    "Technical documentation completeness",
                    "Cross-border automotive data transfers"
                ],
                "specific_requirements": [
                    "Baden-Württemberg technical documentation standards",
                    "Automotive privacy by design implementation",
                    "Stuttgart automotive cluster compliance coordination"
                ],
                "penalty_considerations": [
                    "LfD BW emphasizes process documentation",
                    "Technical measure implementation is heavily weighted",
                    "Documentation gaps result in higher penalties"
                ]
            },
            
            "software_lfd_bw": {
                "enforcement_focus": [
                    "Software development lifecycle privacy integration",
                    "API data protection implementation",
                    "User consent technical implementation"
                ],
                "specific_requirements": [
                    "Stuttgart software cluster compliance standards",
                    "Privacy by design in agile development",
                    "Technical privacy measure documentation"
                ],
                "penalty_considerations": [
                    "LfD BW has high expectations for software companies",
                    "Privacy by design failures result in significant penalties",
                    "Technical implementation quality is critical"
                ]
            },
            
            "manufacturing_ldi_nrw": {
                "enforcement_focus": [
                    "Employee monitoring compliance",
                    "Industrial IoT data protection",
                    "Manufacturing risk management"
                ],
                "specific_requirements": [
                    "NRW manufacturing privacy guidelines",
                    "Industrial data processing risk assessments",
                    "Employee monitoring legal basis documentation"
                ],
                "penalty_considerations": [
                    "LDI NRW emphasizes risk-based compliance",
                    "Employee privacy violations carry higher penalties",
                    "Manufacturing complexity is considered in enforcement"
                ]
            }
        }
    
    def _get_generic_checklist(self, authority: Big4Authority) -> Dict[str, Any]:
        """Generate generic checklist for unknown industry"""
        
        return {
            "industry": "generic",
            "authority": authority.value,
            "template_name": "Generic GDPR Compliance",
            "compliance_complexity": "medium",
            
            "required_documents": [
                {
                    "document_type": "privacy_policy",
                    "name": "Privacy Policy",
                    "description": "General privacy policy covering data processing activities",
                    "compliance_level": "mandatory"
                },
                {
                    "document_type": "data_processing_register",
                    "name": "Data Processing Register (Art. 30)",
                    "description": "Register of processing activities",
                    "compliance_level": "mandatory"
                }
            ],
            
            "critical_areas": [
                {
                    "area": "legal_basis",
                    "description": "Legal basis for data processing (Art. 6)",
                    "risk_level": "high"
                },
                {
                    "area": "data_subject_rights",
                    "description": "Data subject rights implementation (Art. 15-22)",
                    "risk_level": "medium"
                }
            ],
            
            "guidance": {
                "best_practices": [
                    "Implement comprehensive privacy policy",
                    "Establish data subject rights procedures",
                    "Regular compliance monitoring and training"
                ]
            }
        }
    
    def _get_authority_focus_areas(
        self, 
        authority: Big4Authority, 
        template: ComplianceTemplate
    ) -> List[str]:
        """Get authority-specific focus areas for industry"""
        
        authority_focus = {
            Big4Authority.BFDI: [
                "Cross-border data transfers",
                "Large-scale data processing",
                "Federal compliance coordination"
            ],
            Big4Authority.BAYLDA: [
                "Consent management implementation",
                "SME compliance support",
                "Automotive industry focus"
            ],
            Big4Authority.LFD_BW: [
                "Privacy by design implementation",
                "Technical documentation",
                "Software industry compliance"
            ],
            Big4Authority.LDI_NRW: [
                "Risk-based compliance approach",
                "Manufacturing industry focus",
                "Employee data protection"
            ]
        }
        
        return authority_focus.get(authority, ["General GDPR compliance"])
    
    def _get_implementation_priorities(
        self, 
        template: ComplianceTemplate, 
        authority: Big4Authority
    ) -> List[str]:
        """Get implementation priorities based on template and authority"""
        
        priorities = []
        
        # Add high-priority areas from template
        high_priority_areas = [
            area for area in template.critical_compliance_areas
            if area["risk_level"] in ["high", "very_high"]
        ]
        
        for area in high_priority_areas[:3]:  # Top 3 priorities
            priorities.append(f"Implement {area['description']}")
        
        # Add authority-specific priorities
        if authority == Big4Authority.BAYLDA:
            priorities.append("Focus on consent management systems")
        elif authority == Big4Authority.LFD_BW:
            priorities.append("Prioritize technical documentation")
        elif authority == Big4Authority.LDI_NRW:
            priorities.append("Conduct comprehensive risk assessments")
        elif authority == Big4Authority.BFDI:
            priorities.append("Ensure federal compliance standards")
        
        return priorities[:5]  # Limit to top 5 priorities