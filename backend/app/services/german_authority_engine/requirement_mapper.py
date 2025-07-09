# app/services/german_authority_engine/requirement_mapper.py
"""
Requirement Mapper

Maps document content to specific German authority requirements.
Analyzes document text to determine which authority requirements
are addressed, partially addressed, or missing.
"""

import re
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass
import structlog

from app.models.database import Document
from .authority_profiles import AuthorityProfile, AuthorityRequirement

logger = structlog.get_logger()

@dataclass
class RequirementMapping:
    """Mapping between document content and authority requirement"""
    requirement_id: str
    article_reference: str
    coverage_score: float  # 0.0 = not covered, 1.0 = fully covered
    evidence_locations: List[str]  # Document sections that address this requirement
    gaps_identified: List[str]
    confidence_level: float

class RequirementMapper:
    """
    Maps document content to German authority requirements
    
    Analyzes document text using pattern matching, keyword detection,
    and semantic analysis to determine compliance coverage.
    """
    
    def __init__(self):
        # German compliance patterns for requirement detection
        self.requirement_patterns = {
            "Art. 6": {
                "keywords": [
                    "rechtsgrundlage", "legal basis", "rechtmäßigkeit", "lawfulness",
                    "einwilligung", "consent", "vertrag", "contract", "berechtigtes interesse"
                ],
                "required_elements": [
                    "documented_legal_basis", "processing_purpose", "data_categories"
                ],
                "evidence_patterns": [
                    r"rechtsgrundlage.*art\.?\s*6",
                    r"legal basis.*article\s*6",
                    r"einwilligung.*art\.?\s*7",
                    r"berechtigtes interesse"
                ]
            },
            "Art. 13-14": {
                "keywords": [
                    "informationspflicht", "information obligation", "transparenz", "transparency",
                    "datenschutzerklärung", "privacy notice", "betroffene person", "data subject"
                ],
                "required_elements": [
                    "controller_identity", "processing_purposes", "legal_basis", "retention_periods",
                    "data_subject_rights", "contact_information"
                ],
                "evidence_patterns": [
                    r"informationspflicht.*art\.?\s*1[34]",
                    r"datenschutzerklärung",
                    r"privacy notice",
                    r"betroffenenrechte"
                ]
            },
            "Art. 25": {
                "keywords": [
                    "privacy by design", "datenschutz durch technikgestaltung",
                    "privacy by default", "datenschutz-freundliche voreinstellungen",
                    "technical measures", "technische maßnahmen", "organisatorische maßnahmen"
                ],
                "required_elements": [
                    "privacy_by_design_implementation", "default_privacy_settings",
                    "technical_safeguards", "organizational_measures"
                ],
                "evidence_patterns": [
                    r"privacy by design",
                    r"datenschutz durch technikgestaltung",
                    r"privacy by default",
                    r"technische.*maßnahmen"
                ]
            },
            "Art. 30": {
                "keywords": [
                    "verfahrensverzeichnis", "records of processing", "ropa",
                    "verarbeitungstätigkeiten", "processing activities", "dokumentation"
                ],
                "required_elements": [
                    "controller_details", "processing_purposes", "data_categories",
                    "data_recipients", "retention_periods", "security_measures"
                ],
                "evidence_patterns": [
                    r"verfahrensverzeichnis",
                    r"records of processing",
                    r"verarbeitungstätigkeiten",
                    r"art\.?\s*30"
                ]
            },
            "Art. 32": {
                "keywords": [
                    "sicherheit der verarbeitung", "security of processing",
                    "technische maßnahmen", "technical measures", "organisatorische maßnahmen",
                    "verschlüsselung", "encryption", "pseudonymisierung", "pseudonymization"
                ],
                "required_elements": [
                    "technical_security_measures", "organizational_security_measures",
                    "access_controls", "data_protection_measures"
                ],
                "evidence_patterns": [
                    r"sicherheit.*verarbeitung",
                    r"security.*processing",
                    r"verschlüsselung",
                    r"encryption",
                    r"technische.*sicherheitsmaßnahmen"
                ]
            },
            "Art. 35": {
                "keywords": [
                    "datenschutz-folgenabschätzung", "dsfa", "dpia", "impact assessment",
                    "hohes risiko", "high risk", "risikobewertung", "risk assessment"
                ],
                "required_elements": [
                    "systematic_description", "necessity_assessment", "risk_assessment",
                    "mitigation_measures", "consultation_requirements"
                ],
                "evidence_patterns": [
                    r"datenschutz-?folgenabschätzung",
                    r"dsfa",
                    r"dpia",
                    r"impact assessment",
                    r"art\.?\s*35"
                ]
            }
        }
        
        # Industry-specific pattern enhancers
        self.industry_patterns = {
            "automotive": {
                "keywords": [
                    "fahrzeugdaten", "vehicle data", "telematics", "connected car",
                    "fahrverhalten", "driving behavior", "navigation", "standortdaten"
                ],
                "requirements": ["Art. 25", "Art. 32", "Art. 35"]
            },
            "healthcare": {
                "keywords": [
                    "gesundheitsdaten", "health data", "patientendaten", "patient data",
                    "medizinische daten", "medical data", "behandlung", "treatment"
                ],
                "requirements": ["Art. 9", "Art. 32", "Art. 35"]
            },
            "manufacturing": {
                "keywords": [
                    "mitarbeiterdaten", "employee data", "produktionsdaten", "production data",
                    "maschinendaten", "machine data", "industrie 4.0", "iot"
                ],
                "requirements": ["Art. 25", "Art. 30", "Art. 32"]
            }
        }
    
    async def map_documents_to_requirements(
        self,
        documents: List[Document],
        requirements: List[AuthorityRequirement],
        authority_profile: AuthorityProfile
    ) -> Dict[str, Any]:
        """
        Map document content to authority requirements
        
        Returns comprehensive mapping showing which requirements
        are covered, partially covered, or missing.
        """
        logger.info(
            "Starting requirement mapping",
            documents=len(documents),
            requirements=len(requirements),
            authority=authority_profile.authority_id
        )
        
        # Extract all document content
        document_content = await self._extract_document_content(documents)
        
        # Analyze each requirement
        requirement_mappings = {}
        coverage_summary = {
            "fully_covered": [],
            "partially_covered": [],
            "not_covered": [],
            "total_coverage_score": 0.0
        }
        
        for requirement in requirements:
            mapping = await self._analyze_requirement_coverage(
                requirement, document_content, authority_profile
            )
            
            requirement_mappings[requirement.article_reference] = mapping
            
            # Categorize coverage
            if mapping.coverage_score >= 0.8:
                coverage_summary["fully_covered"].append(requirement.article_reference)
            elif mapping.coverage_score >= 0.3:
                coverage_summary["partially_covered"].append(requirement.article_reference)
            else:
                coverage_summary["not_covered"].append(requirement.article_reference)
        
        # Calculate overall coverage
        if requirement_mappings:
            coverage_summary["total_coverage_score"] = sum(
                mapping.coverage_score for mapping in requirement_mappings.values()
            ) / len(requirement_mappings)
        
        logger.info(
            "Requirement mapping completed",
            fully_covered=len(coverage_summary["fully_covered"]),
            partially_covered=len(coverage_summary["partially_covered"]),
            not_covered=len(coverage_summary["not_covered"]),
            total_score=coverage_summary["total_coverage_score"]
        )
        
        return {
            "mappings": requirement_mappings,
            "coverage_summary": coverage_summary,
            "document_analysis": await self._analyze_document_types(documents)
        }
    
    async def _extract_document_content(self, documents: List[Document]) -> Dict[str, str]:
        """Extract searchable content from documents"""
        content = {}
        
        for doc in documents:
            # Get content from chunks if available
            if hasattr(doc, 'chunks') and doc.chunks:
                doc_content = " ".join([
                    chunk.content for chunk in doc.chunks
                    if hasattr(chunk, 'content') and chunk.content
                ])
            else:
                # Fallback to document-level content if available
                doc_content = getattr(doc, 'content', '') or str(doc)
            
            content[doc.filename] = doc_content.lower()
        
        return content
    
    async def _analyze_requirement_coverage(
        self,
        requirement: AuthorityRequirement,
        document_content: Dict[str, str],
        authority_profile: AuthorityProfile
    ) -> RequirementMapping:
        """Analyze how well documents cover a specific requirement"""
        
        article_ref = requirement.article_reference
        patterns = self.requirement_patterns.get(article_ref, {})
        
        # Search for requirement evidence across all documents
        evidence_locations = []
        coverage_scores = []
        gaps_identified = []
        
        all_content = " ".join(document_content.values())
        
        # Keyword matching
        keyword_score = self._calculate_keyword_score(
            all_content, patterns.get("keywords", [])
        )
        
        # Pattern matching
        pattern_score = self._calculate_pattern_score(
            all_content, patterns.get("evidence_patterns", [])
        )
        
        # Required elements check
        elements_score = self._calculate_elements_score(
            all_content, patterns.get("required_elements", [])
        )
        
        # Industry-specific enhancement
        industry_score = self._calculate_industry_score(
            all_content, authority_profile.industry_focus
        )
        
        # Combined coverage score
        coverage_score = (
            keyword_score * 0.3 +
            pattern_score * 0.4 +
            elements_score * 0.2 +
            industry_score * 0.1
        )
        
        # Identify evidence locations
        for filename, content in document_content.items():
            if any(keyword in content for keyword in patterns.get("keywords", [])):
                evidence_locations.append(f"Found in {filename}")
        
        # Identify gaps
        required_elements = patterns.get("required_elements", [])
        for element in required_elements:
            if not self._element_found_in_content(all_content, element):
                gaps_identified.append(f"Missing: {element}")
        
        # Calculate confidence based on evidence strength
        confidence_level = min(1.0, (keyword_score + pattern_score) / 2)
        
        return RequirementMapping(
            requirement_id=requirement.article_reference,
            article_reference=article_ref,
            coverage_score=min(1.0, coverage_score),
            evidence_locations=evidence_locations,
            gaps_identified=gaps_identified,
            confidence_level=confidence_level
        )
    
    def _calculate_keyword_score(self, content: str, keywords: List[str]) -> float:
        """Calculate score based on keyword presence"""
        if not keywords:
            return 0.0
        
        found_keywords = sum(1 for keyword in keywords if keyword.lower() in content)
        return found_keywords / len(keywords)
    
    def _calculate_pattern_score(self, content: str, patterns: List[str]) -> float:
        """Calculate score based on regex pattern matching"""
        if not patterns:
            return 0.0
        
        found_patterns = 0
        for pattern in patterns:
            try:
                if re.search(pattern, content, re.IGNORECASE):
                    found_patterns += 1
            except re.error:
                continue  # Skip invalid patterns
        
        return found_patterns / len(patterns)
    
    def _calculate_elements_score(self, content: str, elements: List[str]) -> float:
        """Calculate score based on required elements presence"""
        if not elements:
            return 0.0
        
        found_elements = sum(
            1 for element in elements
            if self._element_found_in_content(content, element)
        )
        
        return found_elements / len(elements)
    
    def _calculate_industry_score(self, content: str, industry_focus: List[str]) -> float:
        """Calculate bonus score for industry-specific content"""
        industry_bonus = 0.0
        
        for industry in industry_focus:
            industry_lower = industry.lower()
            if industry_lower in self.industry_patterns:
                industry_keywords = self.industry_patterns[industry_lower]["keywords"]
                keyword_matches = sum(
                    1 for keyword in industry_keywords
                    if keyword.lower() in content
                )
                if keyword_matches > 0:
                    industry_bonus += 0.2  # Bonus for industry relevance
        
        return min(1.0, industry_bonus)
    
    def _element_found_in_content(self, content: str, element: str) -> bool:
        """Check if a required element is found in content"""
        element_patterns = {
            "documented_legal_basis": ["rechtsgrundlage", "legal basis", "artikel 6"],
            "processing_purpose": ["zweck", "purpose", "verarbeitungszweck"],
            "data_categories": ["datenkategorien", "data categories", "personenbezogene daten"],
            "controller_identity": ["verantwortlicher", "controller", "kontakt"],
            "retention_periods": ["aufbewahrung", "retention", "speicherdauer"],
            "data_subject_rights": ["betroffenenrechte", "data subject rights", "auskunft"],
            "technical_security_measures": ["technische maßnahmen", "technical measures", "sicherheit"],
            "organizational_security_measures": ["organisatorische maßnahmen", "organizational measures"],
            "privacy_by_design_implementation": ["privacy by design", "datenschutz durch technikgestaltung"],
            "systematic_description": ["systematische beschreibung", "systematic description"],
            "risk_assessment": ["risikobewertung", "risk assessment", "risikoanalyse"]
        }
        
        patterns = element_patterns.get(element, [element])
        return any(pattern.lower() in content for pattern in patterns)
    
    async def _analyze_document_types(self, documents: List[Document]) -> Dict[str, Any]:
        """Analyze document types for completeness assessment"""
        
        doc_types_found = set()
        doc_analysis = {
            "total_documents": len(documents),
            "document_types": [],
            "completeness_indicators": {}
        }
        
        for doc in documents:
            # Extract document type from filename or metadata
            doc_type = self._classify_document_type(doc.filename)
            if doc_type:
                doc_types_found.add(doc_type)
                doc_analysis["document_types"].append({
                    "filename": doc.filename,
                    "type": doc_type,
                    "size": getattr(doc, 'file_size', 0)
                })
        
        # Check completeness against standard compliance documentation
        required_doc_types = {
            "privacy_policy": "Privacy Policy / Datenschutzerklärung",
            "ropa": "Records of Processing / Verfahrensverzeichnis", 
            "dpia": "Data Protection Impact Assessment / DSFA",
            "security_policy": "Security Policy / Sicherheitsrichtlinie",
            "training_materials": "Training Materials / Schulungsunterlagen"
        }
        
        for req_type, description in required_doc_types.items():
            doc_analysis["completeness_indicators"][req_type] = {
                "required": True,
                "found": req_type in doc_types_found,
                "description": description
            }
        
        return doc_analysis
    
    def _classify_document_type(self, filename: str) -> Optional[str]:
        """Classify document type based on filename"""
        filename_lower = filename.lower()
        
        type_patterns = {
            "privacy_policy": ["datenschutz", "privacy", "privacidad"],
            "ropa": ["verfahren", "ropa", "processing", "tätigkeiten"],
            "dpia": ["dsfa", "dpia", "folgenabschätzung", "impact"],
            "security_policy": ["sicherheit", "security", "sicurezza"],
            "training_materials": ["schulung", "training", "awareness"],
            "contract": ["vertrag", "contract", "agreement"],
            "policy": ["richtlinie", "policy", "guideline"]
        }
        
        for doc_type, patterns in type_patterns.items():
            if any(pattern in filename_lower for pattern in patterns):
                return doc_type
        
        return "unknown"