# app/services/compliance_analyzer.py
import time
from typing import List, Dict, Optional, Tuple
from openai import OpenAI

from app.config import settings
from app.models.compliance import *
from app.utils.german_detection import GermanComplianceDetector

class ComplianceAnalyzer:
    """Core compliance analysis using OpenAI with German awareness"""
    
    def __init__(self):
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = settings.openai_model
        self.detector = GermanComplianceDetector()
    
    async def analyze_documents(
        self,
        file_data: List[Tuple[str, str, int]],
        framework: ComplianceFramework = ComplianceFramework.GDPR
    ) -> AnalysisResponse:
        """Analyze documents for compliance with German awareness"""
        
        start_time = time.time()
        
        # Analyze each document
        individual_analyses = []
        for filename, content, size in file_data:
            analysis = await self._analyze_single_document(
                filename, content, size, framework
            )
            individual_analyses.append(analysis)
        
        # Create unified compliance report
        compliance_report = await self._create_compliance_report(
            individual_analyses, framework
        )
        
        processing_time = time.time() - start_time
        
        return AnalysisResponse(
            individual_analyses=individual_analyses,
            compliance_report=compliance_report,
            processing_metadata={
                "processing_time": processing_time,
                "total_documents": len(file_data),
                "total_size": sum(size for _, _, size in file_data),
                "framework": framework.value,
                "german_documents_detected": compliance_report.german_documents_detected
            }
        )
    
    async def _analyze_single_document(
        self,
        filename: str,
        content: str,
        size: int,
        framework: ComplianceFramework
    ) -> DocumentAnalysis:
        """Analyze single document with German legal awareness"""
        
        # Detect language and German terms
        language, confidence = self.detector.detect_language(content, filename)
        german_terms = self.detector.extract_german_terms(content)
        gdpr_articles = self.detector.extract_gdpr_articles(content)
        
        # Create German-aware prompt
        prompt = self._create_analysis_prompt(
            filename, content, framework, language, german_terms
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.2
            )
            
            result = self._parse_analysis_response(
                response.choices[0].message.content,
                filename,
                language,
                size,
                german_terms,
                gdpr_articles
            )
            
            return result
            
        except Exception as e:
            # Fallback analysis if OpenAI fails
            return self._create_fallback_analysis(
                filename, language, size, str(e)
            )
    
    def _create_analysis_prompt(
        self,
        filename: str,
        content: str,
        framework: ComplianceFramework,
        language: str,
        german_terms: Dict[str, List[str]]
    ) -> str:
        """Create German-aware compliance analysis prompt"""
        
        framework_context = self._get_framework_context(framework, language)
        german_context = self._get_german_context(german_terms) if language == "de" else ""
        
        return f"""You are an expert compliance analyst specializing in {framework.value.upper()} for German enterprises.

Document: {filename}
Detected Language: {language}
Content: {content[:2000]}

{framework_context}

{german_context}

IMPORTANT INSTRUCTIONS:
1. If this is a German document, recognize German legal terminology
2. Map German DSGVO terms to international GDPR articles
3. Provide bilingual insights where helpful for German compliance teams
4. Focus on actionable compliance insights, not general summaries

Analyze for:
- Legal compliance requirements and gaps
- Specific regulatory controls and evidence
- Risk indicators and recommendations
- German-specific compliance considerations (if applicable)

Respond with clear, structured analysis focusing on compliance value."""
    
    def _get_framework_context(self, framework: ComplianceFramework, language: str) -> str:
        """Get framework-specific context"""
        
        contexts = {
            ComplianceFramework.GDPR: f"""
GDPR/DSGVO Analysis Focus:
- Legal basis for processing (Article 6)
- Data subject rights (Articles 15-22)
- Privacy by design (Article 25)
- Records of processing (Article 30)
- Security measures (Article 32)
- Data Protection Impact Assessment (Article 35)

{"German Context: Analyze with DSGVO terminology and German supervisory authority requirements." if language == "de" else ""}
""",
            ComplianceFramework.SOC2: """
SOC 2 Trust Service Criteria:
- Security (protection against unauthorized access)
- Availability (system operations and monitoring)
- Processing Integrity (system processing completeness, validity, accuracy)
- Confidentiality (protection of confidential information)
- Privacy (collection, use, retention, disclosure of personal information)
""",
            ComplianceFramework.HIPAA: """
HIPAA Safeguards Analysis:
- Administrative safeguards (workforce training, access management)
- Physical safeguards (facility access, workstation controls, media controls)
- Technical safeguards (access controls, audit controls, encryption)
"""
        }
        
        return contexts.get(framework, "General compliance analysis")
    
    def _get_german_context(self, german_terms: Dict[str, List[str]]) -> str:
        """Generate German-specific context based on detected terms"""
        
        if not german_terms:
            return ""
        
        context = "\nGerman Legal Terms Detected:\n"
        for category, terms in german_terms.items():
            context += f"- {category}: {', '.join(terms)}\n"
        
        context += "\nPlease map these German terms to international compliance standards and provide German-specific recommendations."
        
        return context
    
    def _parse_analysis_response(
        self,
        response_content: str,
        filename: str,
        language: str,
        size: int,
        german_terms: Dict[str, List[str]],
        gdpr_articles: List[str]
    ) -> DocumentAnalysis:
        """Parse OpenAI response into structured analysis"""
        
        # For Day 1, simple text parsing
        # Day 2 will enhance with JSON-structured responses
        
        german_insights = None
        if language == "de" or german_terms:
            german_insights = GermanComplianceInsights(
                dsgvo_articles_found=gdpr_articles,
                german_terms_detected=[
                    term for terms_list in german_terms.values() 
                    for term in terms_list
                ],
                compliance_completeness=0.75  # Placeholder for Day 1
            )
        
        return DocumentAnalysis(
            filename=filename,
            document_language=DocumentLanguage(language),
            compliance_summary=response_content[:500],
            compliance_summary_de=None,  # Day 2 will add bilingual summaries
            control_mappings=[],  # Day 2 will add structured control mapping
            compliance_gaps=["Detailed gap analysis available in Day 2"],
            risk_indicators=["Risk assessment enhanced in Day 2"],
            german_insights=german_insights,
            original_size=size,
            processing_time=1.0  # Placeholder
        )
    
    def _create_fallback_analysis(
        self, filename: str, language: str, size: int, error: str
    ) -> DocumentAnalysis:
        """Create fallback analysis if OpenAI fails"""
        
        return DocumentAnalysis(
            filename=filename,
            document_language=DocumentLanguage(language),
            compliance_summary=f"Analysis failed for {filename}. Error: {error}",
            original_size=size,
            processing_time=0.0
        )
    
    async def _create_compliance_report(
        self,
        analyses: List[DocumentAnalysis],
        framework: ComplianceFramework
    ) -> ComplianceReport:
        """Create unified compliance report"""
        
        german_detected = any(
            analysis.document_language == DocumentLanguage.GERMAN 
            for analysis in analyses
        )
        
        # Calculate basic compliance score
        compliance_score = 0.7  # Placeholder for Day 1
        
        return ComplianceReport(
            framework=framework,
            executive_summary=f"Analyzed {len(analyses)} documents for {framework.value.upper()} compliance.",
            compliance_score=compliance_score,
            documents_analyzed=len(analyses),
            german_documents_detected=german_detected,
            priority_gaps=["Enhanced gap analysis in Day 2"],
            compliance_strengths=["Document analysis completed successfully"],
            next_steps=["Review individual document analyses", "Implement Day 2 enhancements for detailed insights"],
            german_specific_recommendations=["German compliance features enhanced in Day 2"] if german_detected else []
        )