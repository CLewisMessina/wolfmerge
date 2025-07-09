# app/data/german_authorities/state_authorities_south.py
"""
Southern German Data Protection Authorities

Focus: Automotive industry, manufacturing, innovation hubs
Key authorities: BayLDA (Bavaria), LfD BW (Baden-Württemberg)
"""

from typing import Dict
from app.services.german_authority_engine.authority_profiles import (
    GermanAuthority, AuthorityProfile, AuthorityRequirement
)

def get_southern_authorities() -> Dict[GermanAuthority, AuthorityProfile]:
    """Get southern German state data protection authorities"""
    
    return {
        GermanAuthority.BAYERN: AuthorityProfile(
            authority_id="baylda",
            name="Bayerisches Landesamt für Datenschutzaufsicht",
            name_english="Bavarian State Office for Data Protection Supervision",
            jurisdiction="Bavaria - automotive industry, manufacturing, tourism, technology",
            state_code="BY",
            
            # Contact information
            address="Promenade 18, 91522 Ansbach",
            phone="+49 981 180093-0",
            email="poststelle@lda.bayern.de",
            website="https://www.lda.bayern.de",
            
            # Bavaria-specific requirements
            specific_requirements=[
                AuthorityRequirement(
                    article_reference="Art. 32 DSGVO",
                    requirement_text="Enhanced security measures mandatory for automotive data processing including vehicle telematics and connected car privacy",
                    enforcement_priority="high",
                    typical_audit_focus=[
                        "Vehicle data processing and storage",
                        "Connected car privacy protection",
                        "Automotive supplier chain compliance",
                        "Telematics data security measures",
                        "Driver behavior data minimization"
                    ],
                    penalty_range="€5,000 - €15,000,000",
                    industry_specific=["automotive", "manufacturing", "iot", "telematics"],
                    german_legal_reference="DSGVO Art. 32, BayDSG § 24",
                    implementation_guidance="Automotive-specific technical and organizational measures required"
                ),
                
                AuthorityRequirement(
                    article_reference="Art. 25 DSGVO",
                    requirement_text="Privacy by Design mandatory for Bavaria automotive cluster innovations",
                    enforcement_priority="high",
                    typical_audit_focus=[
                        "New vehicle technology privacy assessment",
                        "Automotive R&D data protection",
                        "Connected mobility service design",
                        "Manufacturing process privacy integration"
                    ],
                    penalty_range="€10,000 - €25,000,000",
                    industry_specific=["automotive", "manufacturing", "technology", "innovation"],
                    german_legal_reference="DSGVO Art. 25, BayDSG § 22",
                    implementation_guidance="Automotive innovation must include privacy impact assessment from design phase"
                ),
                
                AuthorityRequirement(
                    article_reference="Art. 35 DSGVO", 
                    requirement_text="DSFA required for high-risk automotive processing including autonomous vehicle data",
                    enforcement_priority="high",
                    typical_audit_focus=[
                        "Autonomous vehicle data processing",
                        "Biometric driver identification systems",
                        "Location tracking and navigation",
                        "Predictive maintenance data collection"
                    ],
                    penalty_range="€15,000 - €30,000,000",
                    industry_specific=["automotive", "autonomous_vehicles", "mobility_services"],
                    german_legal_reference="DSGVO Art. 35, BayDSG § 26",
                    implementation_guidance="Automotive DSFA template available for industry-specific assessments"
                ),
                
                AuthorityRequirement(
                    article_reference="Art. 13-14 DSGVO",
                    requirement_text="Clear information requirements for vehicle occupants about data processing",
                    enforcement_priority="medium",
                    typical_audit_focus=[
                        "In-vehicle privacy notices",
                        "Data processing transparency for passengers",
                        "Rental car privacy information",
                        "Fleet management data disclosure"
                    ],
                    penalty_range="€2,000 - €5,000,000", 
                    industry_specific=["automotive", "rental_services", "fleet_management"],
                    german_legal_reference="DSGVO Art. 13-14, BayDSG § 18",
                    implementation_guidance="Multi-language privacy information required for international vehicle use"
                )
            ],
            
            # Bavaria audit patterns
            audit_patterns={
                "frequency": "Industry-focused campaigns with automotive sector priority, annual manufacturing sweeps",
                "typical_duration": "2-4 months for standard automotive cases, 6+ months for complex autonomous vehicle cases",
                "documentation_requirements": "Technical precision required, German preferred with English summaries for international companies",
                "follow_up_style": "Collaborative technical workshops, industry best practice sharing",
                "sector_specialization": "Deep automotive and manufacturing expertise with technical advisory role",
                "innovation_focus": "Proactive engagement with automotive innovation and Industry 4.0 initiatives"
            },
            
            # Bavaria focus areas
            industry_focus=[
                "Automotive (BMW, Audi, suppliers)",
                "Manufacturing and Industry 4.0",
                "Tourism and hospitality",
                "Technology and innovation",
                "Aerospace and defense",
                "Financial services (Munich)"
            ],
            
            enforcement_style="Collaborative technical approach balancing innovation support with strict compliance requirements",
            penalty_approach="Industry-aware graduated enforcement with technical guidance for automotive innovations",
            
            # Regional coordination
            cross_border_coordination=[
                "Baden-Württemberg LfD (automotive corridor)",
                "Austrian data protection authority",
                "Czech Republic coordination (Škoda)",
                "Federal automotive working groups",
                "EU automotive privacy initiatives"
            ],
            
            # Notable Bayern enforcement examples
            recent_enforcement_examples=[
                "€14.5M fine against H&M for employee surveillance (2020)",
                "Multiple automotive supplier compliance investigations (2022-2023)",
                "Connected car privacy guidance development (2021-2023)",
                "Manufacturing worker monitoring assessments (2022)",
                "Tourism industry COVID-19 data processing reviews (2020-2021)",
                "Automotive telematics privacy enforcement (2023-2024)"
            ],
            
            # Bavaria guidance documents
            guidance_documents=[
                "Automotive Data Protection Guidelines for Bavaria",
                "Connected Vehicle Privacy Assessment Template", 
                "Manufacturing Employee Monitoring Guidelines",
                "Tourism Industry Privacy Best Practices",
                "Industry 4.0 Data Protection Implementation Guide",
                "Autonomous Vehicle Data Processing Framework"
            ],
            
            # Economic and geographic context
            major_cities=["Munich", "Nuremberg", "Augsburg", "Ingolstadt", "Ansbach"],
            key_industries=[
                "Automotive (BMW, Audi, major suppliers)",
                "Manufacturing and mechanical engineering",
                "Information technology",
                "Tourism and hospitality", 
                "Aerospace and defense",
                "Financial services",
                "Energy and utilities"
            ],
            economic_profile="Major automotive and manufacturing hub with strong innovation focus and international business presence"
        ),
        
        GermanAuthority.BADEN_WURTTEMBERG: AuthorityProfile(
            authority_id="lfd_bw",
            name="Der Landesbeauftragte für den Datenschutz und die Informationsfreiheit Baden-Württemberg",
            name_english="State Commissioner for Data Protection and Freedom of Information Baden-Württemberg",
            jurisdiction="Baden-Württemberg - automotive engineering, high-tech manufacturing, innovation",
            state_code="BW",
            
            # Contact information
            address="Lautenschlagerstraße 20, 70173 Stuttgart",
            phone="+49 711 615541-0",
            email="poststelle@lfdi.bwl.de", 
            website="https://www.baden-wuerttemberg.datenschutz.de",
            
            # Baden-Württemberg specific requirements
            specific_requirements=[
                AuthorityRequirement(
                    article_reference="Art. 25 DSGVO",
                    requirement_text="Privacy by Design mandatory for Stuttgart automotive cluster and engineering innovations",
                    enforcement_priority="high",
                    typical_audit_focus=[
                        "Automotive engineering privacy integration",
                        "High-tech manufacturing data protection",
                        "R&D innovation privacy assessment",
                        "Engineering data flows and protection",
                        "Technical system privacy architecture"
                    ],
                    penalty_range="€5,000 - €15,000,000",
                    industry_specific=["automotive", "engineering", "manufacturing", "technology"],
                    german_legal_reference="DSGVO Art. 25, LDSG BW § 4",
                    implementation_guidance="Engineering-focused privacy by design with technical system integration requirements"
                ),
                
                AuthorityRequirement(
                    article_reference="Art. 32 DSGVO",
                    requirement_text="Technical security measures reflecting Baden-Württemberg engineering standards",
                    enforcement_priority="high", 
                    typical_audit_focus=[
                        "Engineering system security architecture",
                        "Manufacturing process data protection",
                        "Automotive component data security",
                        "High-precision manufacturing privacy",
                        "Technical documentation security"
                    ],
                    penalty_range="€8,000 - €20,000,000",
                    industry_specific=["automotive", "engineering", "precision_manufacturing", "technology"],
                    german_legal_reference="DSGVO Art. 32, LDSG BW § 8",
                    implementation_guidance="Engineering-grade technical measures with documented security architecture required"
                ),
                
                AuthorityRequirement(
                    article_reference="Art. 35 DSGVO",
                    requirement_text="DSFA required for high-tech automotive and engineering innovations",
                    enforcement_priority="high",
                    typical_audit_focus=[
                        "Automotive innovation impact assessment", 
                        "Engineering prototype data processing",
                        "Manufacturing system automation privacy",
                        "High-tech research data flows"
                    ],
                    penalty_range="€10,000 - €25,000,000",
                    industry_specific=["automotive", "engineering", "research", "innovation"],
                    german_legal_reference="DSGVO Art. 35, LDSG BW § 12", 
                    implementation_guidance="Technical innovation DSFA template with engineering risk assessment methodology"
                ),
                
                AuthorityRequirement(
                    article_reference="Art. 30 DSGVO",
                    requirement_text="Detailed processing records for complex automotive and engineering operations",
                    enforcement_priority="medium",
                    typical_audit_focus=[
                        "Complex manufacturing process documentation",
                        "Automotive supply chain data flows",
                        "Engineering collaboration data sharing",
                        "Technical system integration records"
                    ],
                    penalty_range="€3,000 - €8,000,000",
                    industry_specific=["automotive", "engineering", "manufacturing", "supply_chain"],
                    german_legal_reference="DSGVO Art. 30, LDSG BW § 6",
                    implementation_guidance="Engineering-focused processing records with technical system documentation"
                )
            ],
            
            # Baden-Württemberg audit patterns
            audit_patterns={
                "frequency": "Innovation-focused with regular automotive and engineering sector assessments",
                "typical_duration": "2-4 months for standard cases, 6+ months for complex engineering assessments",
                "documentation_requirements": "High technical precision, engineering documentation standards, German with technical English summaries",
                "follow_up_style": "Technical deep-dives with engineering focus, collaborative innovation support",
                "technical_expertise": "Strong automotive and engineering technical knowledge in assessment teams",
                "innovation_support": "Proactive privacy guidance for automotive and engineering innovations"
            },
            
            # Focus areas
            industry_focus=[
                "Automotive (Mercedes-Benz, Porsche, Bosch)",
                "Engineering and precision manufacturing", 
                "High-technology and innovation",
                "Information technology and software",
                "Financial services (Stuttgart)",
                "Research and development"
            ],
            
            enforcement_style="Technical precision approach with deep engineering expertise and innovation support",
            penalty_approach="Engineering-aware enforcement with technical guidance and innovation consideration",
            
            # Regional coordination  
            cross_border_coordination=[
                "Bavaria BayLDA (automotive corridor)",
                "French data protection authority (cross-border automotive)",
                "Swiss data protection authority",
                "Federal automotive working groups",
                "EU automotive engineering initiatives"
            ],
            
            # Notable Baden-Württemberg enforcement examples
            recent_enforcement_examples=[
                "Privacy by Design enforcement in automotive sector (2022)",
                "Engineering data protection guidance publication (2023)",
                "Manufacturing worker privacy assessments (2021-2022)",
                "Automotive supplier compliance reviews (2022-2023)",
                "High-tech startup privacy guidance initiatives (2021-2024)",
                "Cross-border automotive data transfer investigations (2023)"
            ],
            
            # Baden-Württemberg guidance documents
            guidance_documents=[
                "Automotive Engineering Privacy by Design Guidelines",
                "Manufacturing Data Protection Implementation Framework",
                "High-Tech Innovation Privacy Assessment Guide",
                "Engineering System Privacy Architecture Standards",
                "Automotive Supply Chain Data Protection Requirements",
                "Technical Documentation Privacy Standards"
            ],
            
            # Economic and geographic context
            major_cities=["Stuttgart", "Karlsruhe", "Mannheim", "Freiburg", "Ulm", "Heilbronn"],
            key_industries=[
                "Automotive (Mercedes-Benz, Porsche, Bosch)",
                "Engineering and precision manufacturing",
                "Information technology and software",
                "Financial services",
                "Research and development",
                "High-technology manufacturing",
                "Renewable energy technology"
            ],
            economic_profile="Premier automotive engineering and high-tech manufacturing region with strong innovation ecosystem and international business focus"
        ),
        
        GermanAuthority.RHEINLAND_PFALZ: AuthorityProfile(
            authority_id="lfd_rlp",
            name="Der Landesbeauftragte für den Datenschutz und die Informationsfreiheit Rheinland-Pfalz",
            name_english="State Commissioner for Data Protection and Freedom of Information Rhineland-Palatinate",
            jurisdiction="Rhineland-Palatinate - chemical industry, logistics, wine industry",
            state_code="RP",
            
            # Contact information
            address="Hintere Bleiche 34, 55116 Mainz",
            phone="+49 6131 208-2449",
            email="poststelle@datenschutz.rlp.de",
            website="https://www.datenschutz.rlp.de",
            
            # Rhineland-Palatinate specific requirements
            specific_requirements=[
                AuthorityRequirement(
                    article_reference="Art. 32 DSGVO",
                    requirement_text="Enhanced security for chemical industry and logistics data processing",
                    enforcement_priority="high",
                    typical_audit_focus=[
                        "Chemical industry safety data protection",
                        "Logistics and transportation data security",
                        "Wine industry customer data protection",
                        "Cross-border logistics data flows"
                    ],
                    penalty_range="€3,000 - €10,000,000",
                    industry_specific=["chemical", "logistics", "agriculture", "transportation"],
                    german_legal_reference="DSGVO Art. 32, LDSG RP § 5",
                    implementation_guidance="Industry-specific security measures for chemical and logistics sectors"
                ),
                
                AuthorityRequirement(
                    article_reference="Art. 6 DSGVO",
                    requirement_text="Clear legal basis documentation for cross-border logistics operations",
                    enforcement_priority="medium",
                    typical_audit_focus=[
                        "International logistics data transfers",
                        "Chemical industry regulatory compliance",
                        "Agricultural data processing justification",
                        "Tourism and hospitality legal basis"
                    ],
                    penalty_range="€2,000 - €5,000,000",
                    industry_specific=["logistics", "chemical", "agriculture", "tourism"],
                    german_legal_reference="DSGVO Art. 6, LDSG RP § 3",
                    implementation_guidance="Documented legal basis required for each cross-border data processing activity"
                )
            ],
            
            audit_patterns={
                "frequency": "Industry-focused with emphasis on chemical and logistics sectors",
                "typical_duration": "2-3 months for standard cases",
                "documentation_requirements": "German documentation with industry-specific compliance focus",
                "follow_up_style": "Collaborative approach with industry expertise",
                "sector_focus": "Chemical industry, logistics, agriculture, and tourism specialization"
            },
            
            industry_focus=["Chemical industry", "Logistics and transportation", "Agriculture and wine", "Tourism"],
            enforcement_style="Industry-collaborative with technical expertise in chemical and logistics sectors",
            penalty_approach="Proportionate enforcement with industry guidance and compliance support",
            
            cross_border_coordination=["Luxembourg", "Belgium", "France", "NRW logistics corridor"],
            recent_enforcement_examples=[
                "Chemical industry data protection assessments (2022-2023)",
                "Logistics sector compliance reviews (2022)",
                "Wine industry customer data protection guidance (2021)"
            ],
            guidance_documents=[
                "Chemical Industry Data Protection Guidelines",
                "Logistics Sector Privacy Requirements",
                "Agriculture and Wine Industry Privacy Guide"
            ],
            
            major_cities=["Mainz", "Ludwigshafen", "Koblenz", "Trier", "Kaiserslautern"],
            key_industries=["Chemical industry", "Logistics", "Agriculture", "Tourism", "Manufacturing"],
            economic_profile="Industrial region with strong chemical industry, international logistics hub, and agricultural/wine production"
        ),
        
        GermanAuthority.SAARLAND: AuthorityProfile(
            authority_id="uld_saarland",
            name="Unabhängiges Datenschutzzentrum Saarland",
            name_english="Independent Data Protection Center Saarland",
            jurisdiction="Saarland - steel industry, cross-border commerce, automotive suppliers",
            state_code="SL",
            
            # Contact information
            address="Fritz-Dobisch-Straße 12, 66111 Saarbrücken",
            phone="+49 681 94781-0",
            email="datenschutzzentrum@datenschutz.saarland.de",
            website="https://www.datenschutz.saarland.de",
            
            # Saarland specific requirements
            specific_requirements=[
                AuthorityRequirement(
                    article_reference="Art. 44-49 DSGVO",
                    requirement_text="Special attention to French cross-border data transfers",
                    enforcement_priority="medium",
                    typical_audit_focus=[
                        "Cross-border worker data processing",
                        "French-German business data flows",
                        "Steel industry international operations",
                        "Automotive supplier cross-border data"
                    ],
                    penalty_range="€2,000 - €8,000,000",
                    industry_specific=["cross_border_business", "steel", "automotive", "manufacturing"],
                    german_legal_reference="DSGVO Kap. V, SDSG § 4",
                    implementation_guidance="Cross-border transfer documentation required for French business operations"
                )
            ],
            
            audit_patterns={
                "frequency": "Cross-border focus with steel and automotive supplier emphasis",
                "typical_duration": "2-3 months",
                "documentation_requirements": "German and French documentation accepted",
                "follow_up_style": "Cross-border coordination with French authorities",
                "specialization": "Cross-border commerce and industrial data protection"
            },
            
            industry_focus=["Steel industry", "Automotive suppliers", "Cross-border commerce", "Manufacturing"],
            enforcement_style="Cross-border collaborative with industrial sector expertise",
            penalty_approach="Proportionate with cross-border business consideration",
            
            cross_border_coordination=["France CNIL", "Luxembourg", "Baden-Württemberg", "Rhineland-Palatinate"],
            recent_enforcement_examples=[
                "Cross-border worker data protection assessments (2021-2022)",
                "Steel industry compliance reviews (2022)"
            ],
            guidance_documents=[
                "Cross-Border Data Protection Guidelines",
                "Steel Industry Privacy Requirements"
            ],
            
            major_cities=["Saarbrücken", "Neunkirchen", "Homburg", "Völklingen"],
            key_industries=["Steel industry", "Automotive suppliers", "Manufacturing", "Cross-border services"],
            economic_profile="Industrial border region with strong French business connections and traditional heavy industry"
        )
    }