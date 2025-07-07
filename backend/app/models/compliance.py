# app/models/compliance.py
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from enum import Enum

class ComplianceFramework(str, Enum):
    GDPR = "gdpr"
    SOC2 = "soc2"
    HIPAA = "hipaa"
    ISO27001 = "iso27001"

class DocumentLanguage(str, Enum):
    GERMAN = "de"
    ENGLISH = "en"
    MIXED = "mixed"
    UNKNOWN = "unknown"

class ComplianceControlMapping(BaseModel):
    control_id: str
    control_name: str
    control_name_de: Optional[str] = None
    evidence_text: str
    confidence: float
    german_legal_reference: Optional[str] = None

class GermanComplianceInsights(BaseModel):
    dsgvo_articles_found: List[str] = []
    german_authority_references: List[str] = []
    compliance_completeness: float = 0.0
    german_terms_detected: List[str] = []

class DocumentAnalysis(BaseModel):
    filename: str
    document_language: DocumentLanguage
    compliance_summary: str
    compliance_summary_de: Optional[str] = None
    control_mappings: List[ComplianceControlMapping] = []
    compliance_gaps: List[str] = []
    risk_indicators: List[str] = []
    german_insights: Optional[GermanComplianceInsights] = None
    original_size: int
    processing_time: float

class ComplianceReport(BaseModel):
    framework: ComplianceFramework
    executive_summary: str
    executive_summary_de: Optional[str] = None
    compliance_score: float
    documents_analyzed: int
    german_documents_detected: bool
    priority_gaps: List[str] = []
    compliance_strengths: List[str] = []
    next_steps: List[str] = []
    german_specific_recommendations: List[str] = []

class AnalysisResponse(BaseModel):
    individual_analyses: List[DocumentAnalysis]
    compliance_report: ComplianceReport
    processing_metadata: Dict[str, Any]