# app/data/german_authorities/state_authorities_north.py
"""
Northern German Data Protection Authorities

Focus: Maritime industry, logistics, media, technology
Key authorities: ULD Schleswig-Holstein, Hamburg BfDI, Bremen LfD, Niedersachsen LfD, MV LfD
"""

from typing import Dict
from app.services.german_authority_engine.authority_profiles import (
    GermanAuthority, AuthorityProfile, AuthorityRequirement
)

def get_northern_authorities() -> Dict[GermanAuthority, AuthorityProfile]:
    """Get northern German state data protection authorities"""
    
    return {
        GermanAuthority.SCHLESWIG_HOLSTEIN: AuthorityProfile(
            authority_id="uld_sh",
            name="Unabhängiges Landeszentrum für Datenschutz Schleswig-Holstein",
            name_english="Independent State Center for Data Protection Schleswig-Holstein",
            jurisdiction="Schleswig-Holstein - technology, maritime, wind energy",
            state_code="SH",
            
            address="Holstenstraße 98, 24103 Kiel",
            phone="+49 431 988-1200",
            email="mail@datenschutzzentrum.de",
            website="https://www.datenschutzzentrum.de",
            
            specific_requirements=[
                AuthorityRequirement(
                    article_reference="Art. 25 DSGVO",
                    requirement_text="Privacy by Design mandatory for technology innovations and maritime data processing",
                    enforcement_priority="high",
                    typical_audit_focus=[
                        "Technology startup privacy implementation",
                        "Maritime data processing compliance",
                        "Wind energy data collection",
                        "Digital innovation privacy integration"
                    ],
                    penalty_range="€5,000 - €12,000,000",
                    industry_specific=["technology", "maritime", "energy", "innovation"],
                    german_legal_reference="DSGVO Art. 25, LDSG SH § 4",
                    implementation_guidance="Technology-focused privacy by design with innovation support"
                ),
                
                AuthorityRequirement(
                    article_reference="Art. 32 DSGVO",
                    requirement_text="Enhanced security for maritime and offshore data processing",
                    enforcement_priority="medium",
                    typical_audit_focus=[
                        "Maritime communication security",
                        "Offshore wind farm data protection",
                        "Port logistics data security",
                        "Shipping data processing"
                    ],
                    penalty_range="€3,000 - €8,000,000",
                    industry_specific=["maritime", "logistics", "energy"],
                    german_legal_reference="DSGVO Art. 32, LDSG SH § 6"
                )
            ],
            
            audit_patterns={
                "frequency": "Technology and innovation focused with maritime sector attention",
                "typical_duration": "2-3 months",
                "documentation_requirements": "Technology-focused documentation, German preferred",
                "follow_up_style": "Innovation-supportive with technical guidance",
                "sector_expertise": "Strong technology and maritime industry knowledge"
            },
            
            industry_focus=["Technology and innovation", "Maritime industry", "Wind energy", "Digital services"],
            enforcement_style="Innovation-supportive with technology expertise and collaborative approach",
            penalty_approach="Graduated enforcement with innovation consideration and technical guidance",
            
            cross_border_coordination=["Denmark", "Hamburg", "Niedersachsen", "Maritime authorities"],
            recent_enforcement_examples=[
                "Technology startup privacy guidance initiatives (2021-2023)",
                "Maritime data protection assessments (2022)",
                "Wind energy data collection compliance reviews (2023)"
            ],
            guidance_documents=[
                "Technology Innovation Privacy Guidelines",
                "Maritime Data Protection Framework",
                "Digital Services Privacy Requirements"
            ],
            
            major_cities=["Kiel", "Lübeck", "Flensburg", "Neumünster"],
            key_industries=["Technology and software", "Maritime and shipping", "Wind energy", "Tourism"],
            economic_profile="Technology and maritime hub with strong innovation ecosystem and renewable energy focus"
        ),
        
        GermanAuthority.HAMBURG: AuthorityProfile(
            authority_id="bfdi_hamburg",
            name="Hamburgischer Beauftragte für Datenschutz und Informationsfreiheit", 
            name_english="Hamburg Commissioner for Data Protection and Freedom of Information",
            jurisdiction="Hamburg - media, logistics, maritime trade, international business",
            state_code="HH",
            
            address="Klosterwall 6, 20095 Hamburg",
            phone="+49 40 428 54-4040",
            email="mailbox@datenschutz.hamburg.de",
            website="https://datenschutz-hamburg.de",
            
            specific_requirements=[
                AuthorityRequirement(
                    article_reference="Art. 44-49 DSGVO",
                    requirement_text="Enhanced requirements for international trade and maritime data transfers",
                    enforcement_priority="high",
                    typical_audit_focus=[
                        "International shipping data transfers",
                        "Port and logistics data flows",
                        "Media and broadcasting data processing",
                        "International business data compliance"
                    ],
                    penalty_range="€8,000 - €15,000,000",
                    industry_specific=["logistics", "maritime", "media", "international_business"],
                    german_legal_reference="DSGVO Kap. V, HmbDSG § 5",
                    implementation_guidance="International business focus with maritime trade expertise"
                ),
                
                AuthorityRequirement(
                    article_reference="Art. 30 DSGVO",
                    requirement_text="Comprehensive processing records for complex logistics and media operations",
                    enforcement_priority="medium",
                    typical_audit_focus=[
                        "Complex logistics process documentation",
                        "Media content processing records",
                        "International trade data flows",
                        "Port operations data management"
                    ],
                    penalty_range="€5,000 - €10,000,000",
                    industry_specific=["logistics", "media", "maritime", "trade"],
                    german_legal_reference="DSGVO Art. 30, HmbDSG § 7"
                )
            ],
            
            audit_patterns={
                "frequency": "International business and media focused with port operations priority",
                "typical_duration": "3-4 months for complex international cases",
                "documentation_requirements": "International business documentation, German/English",
                "follow_up_style": "Business-focused with international trade expertise",
                "specialization": "Media, logistics, and international business expertise"
            },
            
            industry_focus=["Media and broadcasting", "Logistics and trade", "Maritime industry", "International business"],
            enforcement_style="Business-pragmatic with international expertise and media industry knowledge",
            penalty_approach="Business-aware enforcement with international trade consideration",
            
            cross_border_coordination=["International shipping authorities", "Media regulators", "Schleswig-Holstein", "Niedersachsen"],
            recent_enforcement_examples=[
                "International shipping data compliance reviews (2022-2023)",
                "Media broadcasting privacy assessments (2021-2022)",
                "Port logistics data protection compliance (2023)"
            ],
            guidance_documents=[
                "International Business Data Protection Guide",
                "Media Industry Privacy Requirements",
                "Maritime Trade Data Compliance Framework"
            ],
            
            major_cities=["Hamburg"],
            key_industries=["Media and broadcasting", "Logistics and shipping", "International trade", "Aviation"],
            economic_profile="Major international trade and media hub with significant maritime and logistics operations"
        ),
        
        GermanAuthority.BREMEN: AuthorityProfile(
            authority_id="lfd_bremen",
            name="Landesbeauftragte für Datenschutz und Informationsfreiheit Bremen",
            name_english="State Commissioner for Data Protection and Freedom of Information Bremen",
            jurisdiction="Bremen - aerospace, logistics, automotive suppliers, maritime",
            state_code="HB",
            
            address="Arbergen 1, 28325 Bremen",
            phone="+49 421 361-2010",
            email="office@datenschutz.bremen.de",
            website="https://www.datenschutz.bremen.de",
            
            specific_requirements=[
                AuthorityRequirement(
                    article_reference="Art. 32 DSGVO",
                    requirement_text="Enhanced security for aerospace and automotive supplier data processing",
                    enforcement_priority="high",
                    typical_audit_focus=[
                        "Aerospace industry data security",
                        "Automotive supplier compliance",
                        "Logistics data protection",
                        "Maritime trade security"
                    ],
                    penalty_range="€4,000 - €10,000,000",
                    industry_specific=["aerospace", "automotive", "logistics", "maritime"],
                    german_legal_reference="DSGVO Art. 32, BremDSG § 5"
                )
            ],
            
            audit_patterns={
                "frequency": "Industry-focused with aerospace and automotive priority",
                "typical_duration": "2-3 months",
                "documentation_requirements": "Industry-specific documentation, German",
                "follow_up_style": "Technical industry expertise with compliance support"
            },
            
            industry_focus=["Aerospace", "Automotive suppliers", "Logistics", "Maritime"],
            enforcement_style="Industry-collaborative with technical expertise",
            penalty_approach="Proportionate with industry guidance",
            
            cross_border_coordination=["Niedersachsen", "Hamburg", "Aerospace authorities"],
            recent_enforcement_examples=[
                "Aerospace industry compliance assessments (2022)",
                "Automotive supplier reviews (2023)"
            ],
            guidance_documents=[
                "Aerospace Data Protection Guidelines",
                "Automotive Supplier Compliance Framework"
            ],
            
            major_cities=["Bremen", "Bremerhaven"],
            key_industries=["Aerospace", "Automotive suppliers", "Logistics", "Maritime trade"],
            economic_profile="Aerospace and automotive supplier hub with major port and logistics operations"
        ),
        
        GermanAuthority.NIEDERSACHSEN: AuthorityProfile(
            authority_id="lfd_niedersachsen",
            name="Landesbeauftragte für den Datenschutz Niedersachsen",
            name_english="State Commissioner for Data Protection Lower Saxony",
            jurisdiction="Niedersachsen - automotive, energy, agriculture, manufacturing",
            state_code="NI",
            
            address="Prinzenstraße 5, 30159 Hannover",
            phone="+49 511 120-4500",
            email="poststelle@lfd.niedersachsen.de",
            website="https://lfd.niedersachsen.de",
            
            specific_requirements=[
                AuthorityRequirement(
                    article_reference="Art. 25 DSGVO",
                    requirement_text="Privacy by Design for automotive and energy sector innovations",
                    enforcement_priority="high",
                    typical_audit_focus=[
                        "Automotive manufacturing privacy",
                        "Energy sector data processing",
                        "Agricultural data protection",
                        "Manufacturing process privacy"
                    ],
                    penalty_range="€5,000 - €12,000,000",
                    industry_specific=["automotive", "energy", "agriculture", "manufacturing"],
                    german_legal_reference="DSGVO Art. 25, NDSG § 4"
                ),
                
                AuthorityRequirement(
                    article_reference="Art. 35 DSGVO",
                    requirement_text="DSFA required for high-risk automotive and energy processing",
                    enforcement_priority="medium",
                    typical_audit_focus=[
                        "Automotive production data assessment",
                        "Energy grid data processing",
                        "Agricultural sensor data collection",
                        "Manufacturing automation privacy"
                    ],
                    penalty_range="€6,000 - €15,000,000",
                    industry_specific=["automotive", "energy", "agriculture"],
                    german_legal_reference="DSGVO Art. 35, NDSG § 8"
                )
            ],
            
            audit_patterns={
                "frequency": "Automotive and energy sector focused with agricultural attention",
                "typical_duration": "3-4 months for complex industrial cases",
                "documentation_requirements": "Industrial documentation, German preferred",
                "follow_up_style": "Industry-supportive with technical guidance",
                "sector_expertise": "Strong automotive, energy, and agricultural knowledge"
            },
            
            industry_focus=["Automotive (Volkswagen)", "Energy and utilities", "Agriculture", "Manufacturing"],
            enforcement_style="Industry-collaborative with technical expertise and innovation support",
            penalty_approach="Industry-aware graduated enforcement with technical guidance",
            
            cross_border_coordination=["Bremen", "Hamburg", "Schleswig-Holstein", "NRW automotive corridor"],
            recent_enforcement_examples=[
                "Automotive manufacturing privacy assessments (2022-2023)",
                "Energy sector data protection reviews (2022)",
                "Agricultural data collection compliance (2023)"
            ],
            guidance_documents=[
                "Automotive Manufacturing Privacy Guidelines",
                "Energy Sector Data Protection Framework",
                "Agricultural Data Collection Best Practices"
            ],
            
            major_cities=["Hannover", "Braunschweig", "Oldenburg", "Osnabrück", "Wolfsburg"],
            key_industries=["Automotive (Volkswagen)", "Energy and utilities", "Agriculture", "Manufacturing", "Aerospace"],
            economic_profile="Major automotive manufacturing region with significant energy sector and agricultural innovation"
        ),
        
        GermanAuthority.MECKLENBURG_VORPOMMERN: AuthorityProfile(
            authority_id="lfd_mv",
            name="Landesbeauftragte für Datenschutz und Informationsfreiheit Mecklenburg-Vorpommern",
            name_english="State Commissioner for Data Protection and Freedom of Information Mecklenburg-Western Pomerania",
            jurisdiction="Mecklenburg-Vorpommern - tourism, maritime, energy, agriculture",
            state_code="MV",
            
            address="Lennéstraße 1, 19053 Schwerin",
            phone="+49 385 59494-0",
            email="info@datenschutz-mv.de", 
            website="https://www.datenschutz-mv.de",
            
            specific_requirements=[
                AuthorityRequirement(
                    article_reference="Art. 6 DSGVO",
                    requirement_text="Clear legal basis for tourism and maritime data processing",
                    enforcement_priority="medium",
                    typical_audit_focus=[
                        "Tourism customer data processing",
                        "Maritime data collection",
                        "Energy sector compliance",
                        "Agricultural data processing"
                    ],
                    penalty_range="€2,000 - €6,000,000",
                    industry_specific=["tourism", "maritime", "energy", "agriculture"],
                    german_legal_reference="DSGVO Art. 6, DSG M-V § 3"
                )
            ],
            
            audit_patterns={
                "frequency": "Tourism and maritime focused with seasonal attention",
                "typical_duration": "2-3 months",
                "documentation_requirements": "German documentation with tourism focus",
                "follow_up_style": "Tourism-aware with maritime expertise"
            },
            
            industry_focus=["Tourism and hospitality", "Maritime industry", "Energy (wind/offshore)", "Agriculture"],
            enforcement_style="Tourism-supportive with maritime and energy expertise",
            penalty_approach="Proportionate with seasonal business consideration",
            
            cross_border_coordination=["Poland", "Schleswig-Holstein", "Brandenburg", "Maritime authorities"],
            recent_enforcement_examples=[
                "Tourism data protection assessments (2021-2022)",
                "Maritime compliance reviews (2022)",
                "Energy sector guidance (2023)"
            ],
            guidance_documents=[
                "Tourism Industry Data Protection Guide",
                "Maritime Data Processing Framework",
                "Energy Sector Privacy Requirements"
            ],
            
            major_cities=["Schwerin", "Rostock", "Stralsund", "Greifswald"],
            key_industries=["Tourism and hospitality", "Maritime and shipping", "Wind energy", "Agriculture"],
            economic_profile="Tourism and maritime region with growing renewable energy sector and agricultural base"
        )
    }