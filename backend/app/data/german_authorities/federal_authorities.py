# app/data/german_authorities/federal_authorities.py
"""
Federal German Data Protection Authorities

Primary focus: BfDI (Bundesbeauftragte für den Datenschutz und die Informationsfreiheit)
Handles: International transfers, cross-border cases, federal government processing
"""

from typing import Dict
from app.services.german_authority_engine.authority_profiles import (
    GermanAuthority, AuthorityProfile, AuthorityRequirement
)

def get_federal_authorities() -> Dict[GermanAuthority, AuthorityProfile]:
    """Get federal level German data protection authorities"""
    
    return {
        GermanAuthority.BFDI: AuthorityProfile(
            authority_id="bfdi",
            name="Bundesbeauftragte für den Datenschutz und die Informationsfreiheit",
            name_english="Federal Commissioner for Data Protection and Freedom of Information",
            jurisdiction="Federal level, international transfers, cross-border processing",
            state_code="DE",
            
            # Contact information
            address="Graurheindorfer Str. 153, 53117 Bonn",
            phone="+49 228 997799-0",
            email="poststelle@bfdi.bund.de", 
            website="https://www.bfdi.bund.de",
            
            # Federal-specific requirements
            specific_requirements=[
                AuthorityRequirement(
                    article_reference="Art. 30 DSGVO",
                    requirement_text="Verfahrensverzeichnis must include detailed cross-border transfer documentation with adequacy decisions and SCCs",
                    enforcement_priority="high",
                    typical_audit_focus=[
                        "International data transfers",
                        "Adequacy decisions compliance", 
                        "Standard Contractual Clauses (SCCs)",
                        "Binding Corporate Rules (BCRs)",
                        "Third country transfer safeguards"
                    ],
                    penalty_range="€10,000 - €20,000,000",
                    industry_specific=["multinational_corporations", "cloud_providers", "international_business"],
                    german_legal_reference="DSGVO Art. 30, BDSG § 70",
                    implementation_guidance="Detailed transfer documentation required for each third country data flow"
                ),
                
                AuthorityRequirement(
                    article_reference="Art. 35 DSGVO",
                    requirement_text="DSFA mandatory for federal government processing and high-risk international transfers",
                    enforcement_priority="high",
                    typical_audit_focus=[
                        "Government contract processing",
                        "Public sector data flows",
                        "International transfer risk assessment",
                        "Cross-border law enforcement cooperation"
                    ],
                    penalty_range="€5,000 - €10,000,000",
                    industry_specific=["government_contractors", "public_services", "law_enforcement"],
                    german_legal_reference="DSGVO Art. 35, BDSG § 67",
                    implementation_guidance="Federal DSFA template must be used for government-related processing"
                ),
                
                AuthorityRequirement(
                    article_reference="Art. 44-49 DSGVO",
                    requirement_text="Enhanced safeguards required for international transfers to non-adequate countries",
                    enforcement_priority="high",
                    typical_audit_focus=[
                        "Transfer mechanisms (SCCs, BCRs, derogations)",
                        "Risk assessment for target countries",
                        "Technical and organizational measures",
                        "Data subject notification requirements"
                    ],
                    penalty_range="€25,000 - €50,000,000",
                    industry_specific=["global_corporations", "cloud_services", "international_platforms"],
                    german_legal_reference="DSGVO Kap. V, BDSG § 85-88",
                    implementation_guidance="Pre-transfer assessment mandatory for each non-EU transfer"
                ),
                
                AuthorityRequirement(
                    article_reference="Art. 83 DSGVO", 
                    requirement_text="Federal coordination role in cross-border enforcement cases",
                    enforcement_priority="medium",
                    typical_audit_focus=[
                        "Lead supervisory authority determination",
                        "Cross-border case coordination",
                        "Consistency mechanism compliance",
                        "Joint enforcement actions"
                    ],
                    penalty_range="€50,000 - €100,000,000",
                    industry_specific=["large_corporations", "digital_platforms", "multinational_services"],
                    german_legal_reference="DSGVO Art. 83, BDSG § 41-43",
                    implementation_guidance="Coordinate with relevant state authorities for multi-jurisdictional cases"
                )
            ],
            
            # Federal audit patterns
            audit_patterns={
                "frequency": "Risk-based approach focusing on international transfers and cross-border complaints",
                "typical_duration": "3-6 months for standard cases, 12+ months for complex international cases",
                "documentation_requirements": "Comprehensive bilingual documentation (German/English), legal precision required",
                "follow_up_style": "Formal written responses with legal justification, international coordination",
                "international_cooperation": "Active participant in EDPB consistency mechanism and bilateral agreements",
                "technical_focus": "High technical standards for transfer mechanisms and security measures"
            },
            
            # Focus areas
            industry_focus=[
                "International business",
                "Government and public sector", 
                "Large multinational corporations",
                "Cloud service providers",
                "Digital platforms",
                "Financial services with international operations"
            ],
            
            enforcement_style="Formal legal approach with emphasis on regulatory consistency and international coordination",
            penalty_approach="Severe penalties for international transfer violations, graduated approach for domestic issues",
            
            # Federal coordination
            cross_border_coordination=[
                "European Data Protection Board (EDPB)",
                "All 16 German state authorities",
                "International enforcement cooperation",
                "EU-US Privacy Shield successor mechanisms",
                "Bilateral data protection agreements"
            ],
            
            # Notable enforcement examples
            recent_enforcement_examples=[
                "€35M fine against Meta for WhatsApp data processing coordination (2021)",
                "€746M fine coordination against Amazon for international data flows (2021)",
                "Multiple COVID-19 contact tracing app assessments (2020-2022)",
                "Schrems II decision implementation guidance (2020-2023)",
                "Digital Services Act coordination for large platforms (2023-2024)"
            ],
            
            # Federal guidance documents
            guidance_documents=[
                "International Transfer Guidelines (Schrems II Implementation)",
                "Federal Government DSFA Template",
                "Cloud Service Provider Assessment Guide", 
                "Cross-border Enforcement Cooperation Procedures",
                "Digital Sovereignty and Data Protection Guidelines",
                "AI and Automated Decision-Making Assessment Framework"
            ],
            
            # Geographic and economic context
            major_cities=["Bonn (headquarters)", "Berlin (liaison office)"],
            key_industries=[
                "Federal government and administration",
                "International business headquarters", 
                "Technology and digital services",
                "Financial services",
                "Telecommunications",
                "Defense and security contractors"
            ],
            economic_profile="Federal oversight jurisdiction covering international business operations and government sector across all German states"
        )
    }