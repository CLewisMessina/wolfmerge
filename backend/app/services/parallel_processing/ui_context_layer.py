# app/services/parallel_processing/ui_context_layer.py
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
import structlog

from .job_queue import DocumentJob
from ..docling_processor import DoclingProcessor

logger = structlog.get_logger()

class AuditScenario(Enum):
    """Detected audit/compliance scenarios for UI automation"""
    AUDIT_PREPARATION = "audit_preparation"
    NEW_SERVICE_LAUNCH = "new_service_launch"
    INCIDENT_RESPONSE = "incident_response"
    POLICY_REVIEW = "policy_review"
    VENDOR_ASSESSMENT = "vendor_assessment"
    COMPLIANCE_GAP_ANALYSIS = "compliance_gap_analysis"
    UNKNOWN = "unknown"

class IndustryType(Enum):
    """Detected industry types"""
    AUTOMOTIVE = "automotive"
    HEALTHCARE = "healthcare"
    MANUFACTURING = "manufacturing"
    FINANCIAL = "financial"
    TECHNOLOGY = "technology"
    RETAIL = "retail"
    UNKNOWN = "unknown"

class GermanAuthority(Enum):
    """German data protection authorities"""
    BFDI = "bfdi"  # Federal
    BAYLDA = "baylda"  # Bavaria
    LFD_BW = "lfd_bw"  # Baden-Württemberg
    BERLIN_BFDI = "berlin_bfdi"
    UNKNOWN = "unknown"

@dataclass
class SmartAction:
    """Smart action suggestion for UI"""
    action_id: str
    label: str
    description: str
    priority: str  # high, medium, low
    category: str  # generate, fix, export, analyze
    endpoint: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    estimated_time: int = 30  # seconds
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_id": self.action_id,
            "label": self.label,
            "description": self.description,
            "priority": self.priority,
            "category": self.category,
            "endpoint": self.endpoint,
            "parameters": self.parameters,
            "estimated_time": self.estimated_time
        }

@dataclass
class UIContext:
    """Comprehensive UI context for smart interface"""
    # Scenario detection
    detected_scenario: AuditScenario
    scenario_confidence: float
    scenario_description: str
    
    # Industry and authority detection
    industry_detected: IndustryType
    industry_confidence: float
    german_authority: GermanAuthority
    
    # Document intelligence
    document_types: List[str]
    compliance_completeness: float
    missing_document_types: List[str]
    
    # Smart suggestions
    suggested_actions: List[SmartAction]
    priority_risks: List[str]
    quick_wins: List[str]
    
    # Portfolio insights
    portfolio_score: float
    german_content_percentage: float
    total_documents: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario": {
                "type": self.detected_scenario.value,
                "confidence": self.scenario_confidence,
                "description": self.scenario_description
            },
            "industry": {
                "type": self.industry_detected.value,
                "confidence": self.industry_confidence
            },
            "german_authority": self.german_authority.value,
            "document_intelligence": {
                "types_found": self.document_types,
                "completeness": self.compliance_completeness,
                "missing_types": self.missing_document_types
            },
            "smart_actions": [action.to_dict() for action in self.suggested_actions],
            "insights": {
                "priority_risks": self.priority_risks,
                "quick_wins": self.quick_wins,
                "portfolio_score": self.portfolio_score,
                "german_content_percentage": self.german_content_percentage,
                "total_documents": self.total_documents
            }
        }

class UIContextLayer:
    """Intelligent UI context detection for zero-friction compliance analysis"""
    
    def __init__(self):
        # Document type patterns for detection
        self.document_patterns = {
            'privacy_policy': [
                'privacy policy', 'datenschutzerklärung', 'data protection notice',
                'privacy notice', 'datenschutzrichtlinie'
            ],
            'dsfa': [
                'dsfa', 'dpia', 'data protection impact assessment',
                'datenschutz-folgenabschätzung', 'folgenabschätzung'
            ],
            'ropa': [
                'ropa', 'records of processing', 'verfahrensverzeichnis',
                'processing activities', 'verarbeitungstätigkeiten'
            ],
            'consent_form': [
                'consent', 'einwilligung', 'opt-in', 'zustimmung',
                'consent form', 'einwilligungserklärung'
            ],
            'contract': [
                'contract', 'vertrag', 'agreement', 'vereinbarung',
                'data processing agreement', 'auftragsverarbeitungsvertrag'
            ],
            'breach_response': [
                'breach', 'incident', 'datenpanne', 'sicherheitsvorfall',
                'breach response', 'incident response'
            ],
            'training_material': [
                'training', 'schulung', 'awareness', 'sensibilisierung',
                'employee training', 'mitarbeiterschulung'
            ],
            'audit_document': [
                'audit', 'prüfung', 'compliance report', 'compliance-bericht',
                'assessment', 'bewertung'
            ]
        }
        
        # Industry detection patterns
        self.industry_patterns = {
            IndustryType.AUTOMOTIVE: [
                'fahrzeug', 'automotive', 'car', 'vehicle', 'kfz',
                'bmw', 'mercedes', 'audi', 'porsche', 'volkswagen',
                'connected car', 'telematics', 'kba'
            ],
            IndustryType.HEALTHCARE: [
                'patient', 'medical', 'health', 'gesundheit', 'kranken',
                'hospital', 'clinic', 'arzt', 'medicine', 'pharma',
                'bfarm', 'medical device', 'clinical trial'
            ],
            IndustryType.MANUFACTURING: [
                'manufacturing', 'production', 'factory', 'plant',
                'industrie', 'fertigung', 'produktion', 'maschinenbau',
                'industry 4.0', 'iot', 'sensor'
            ],
            IndustryType.FINANCIAL: [
                'bank', 'financial', 'insurance', 'finanz', 'versicherung',
                'bafin', 'payment', 'credit', 'loan', 'investment'
            ],
            IndustryType.TECHNOLOGY: [
                'software', 'technology', 'tech', 'digital', 'platform',
                'saas', 'cloud', 'api', 'data processing'
            ],
            IndustryType.RETAIL: [
                'retail', 'shop', 'store', 'customer', 'ecommerce',
                'online shop', 'marketplace', 'consumer'
            ]
        }
        
        # Scenario detection patterns
        self.scenario_patterns = {
            AuditScenario.AUDIT_PREPARATION: [
                'audit', 'prüfung', 'inspection', 'inspektion',
                'compliance check', 'regulatory review',
                'authority visit', 'behördliche prüfung'
            ],
            AuditScenario.NEW_SERVICE_LAUNCH: [
                'new service', 'product launch', 'neuer service',
                'go-live', 'rollout', 'implementation',
                'service introduction'
            ],
            AuditScenario.INCIDENT_RESPONSE: [
                'breach', 'incident', 'security event', 'datenpanne',
                'data leak', 'unauthorized access', 'cyberattack'
            ],
            AuditScenario.POLICY_REVIEW: [
                'policy review', 'richtlinien überprüfung',
                'policy update', 'revision', 'überarbeitung'
            ],
            AuditScenario.VENDOR_ASSESSMENT: [
                'vendor', 'supplier', 'third party', 'lieferant',
                'service provider', 'contractor', 'partner'
            ]
        }
        
        # German authority detection
        self.authority_patterns = {
            GermanAuthority.BFDI: [
                'bfdi', 'bundesbeauftragte', 'federal', 'international transfer',
                'cross-border', 'grenzüberschreitend'
            ],
            GermanAuthority.BAYLDA: [
                'baylda', 'bayern', 'bavaria', 'munich', 'münchen',
                'automotive', 'bmw', 'audi'
            ],
            GermanAuthority.LFD_BW: [
                'lfd', 'baden-württemberg', 'stuttgart', 'mercedes',
                'porsche', 'engineering'
            ],
            GermanAuthority.BERLIN_BFDI: [
                'berlin', 'hauptstadt', 'government', 'regierung'
            ]
        }
    
    def analyze_ui_context(self, jobs: List[DocumentJob]) -> UIContext:
        """Analyze document jobs to create intelligent UI context"""
        
        logger.info(
            "Analyzing UI context",
            job_count=len(jobs),
            german_jobs=sum(1 for job in jobs if job.is_german_compliance)
        )
        
        # Detect scenario
        detected_scenario, scenario_confidence = self._detect_scenario(jobs)
        
        # Detect industry
        industry_type, industry_confidence = self._detect_industry(jobs)
        
        # Detect German authority
        german_authority = self._detect_german_authority(jobs, industry_type)
        
        # Analyze document portfolio
        document_types = self._analyze_document_types(jobs)
        completeness, missing_types = self._assess_completeness(document_types, detected_scenario)
        
        # Generate smart actions
        smart_actions = self._generate_smart_actions(
            detected_scenario, document_types, missing_types, industry_type
        )
        
        # Calculate portfolio insights
        portfolio_score = self._calculate_portfolio_score(jobs, document_types)
        german_percentage = sum(1 for job in jobs if job.is_german_compliance) / len(jobs) * 100
        
        # Identify priority risks and quick wins
        priority_risks = self._identify_priority_risks(jobs, missing_types)
        quick_wins = self._identify_quick_wins(jobs, document_types)
        
        scenario_description = self._generate_scenario_description(
            detected_scenario, industry_type, len(jobs)
        )
        
        context = UIContext(
            detected_scenario=detected_scenario,
            scenario_confidence=scenario_confidence,
            scenario_description=scenario_description,
            industry_detected=industry_type,
            industry_confidence=industry_confidence,
            german_authority=german_authority,
            document_types=document_types,
            compliance_completeness=completeness,
            missing_document_types=missing_types,
            suggested_actions=smart_actions,
            priority_risks=priority_risks,
            quick_wins=quick_wins,
            portfolio_score=portfolio_score,
            german_content_percentage=german_percentage,
            total_documents=len(jobs)
        )
        
        logger.info(
            "UI context analysis completed",
            scenario=detected_scenario.value,
            industry=industry_type.value,
            authority=german_authority.value,
            portfolio_score=portfolio_score,
            smart_actions_count=len(smart_actions)
        )
        
        return context
    
    def _detect_scenario(self, jobs: List[DocumentJob]) -> tuple[AuditScenario, float]:
        """Detect the most likely compliance scenario"""
        
        scenario_scores = {scenario: 0.0 for scenario in AuditScenario}
        
        for job in jobs:
            content_preview = ""
            try:
                content_preview = job.content.decode('utf-8', errors='ignore')[:1000].lower()
            except:
                pass
            
            filename_lower = job.filename.lower()
            all_text = content_preview + " " + filename_lower
            
            # Score each scenario based on pattern matches
            for scenario, patterns in self.scenario_patterns.items():
                for pattern in patterns:
                    if pattern in all_text:
                        scenario_scores[scenario] += 1.0
                        if job.is_german_compliance:
                            scenario_scores[scenario] += 0.5  # Bonus for German docs
        
        # Special scenario detection logic
        doc_types = [indicator for job in jobs for indicator in job.compliance_indicators]
        
        # Audit preparation: comprehensive document set
        if len(set(doc_types)) >= 4:
            scenario_scores[AuditScenario.AUDIT_PREPARATION] += 2.0
        
        # Policy review: multiple policy documents
        policy_indicators = ['privacy_policy', 'policy', 'richtlinie']
        if sum(1 for indicator in doc_types if any(p in indicator for p in policy_indicators)) >= 2:
            scenario_scores[AuditScenario.POLICY_REVIEW] += 1.5
        
        # Find highest scoring scenario
        if not any(scenario_scores.values()):
            return AuditScenario.UNKNOWN, 0.0
        
        best_scenario = max(scenario_scores.items(), key=lambda x: x[1])
        confidence = min(1.0, best_scenario[1] / len(jobs))
        
        return best_scenario[0], confidence
    
    def _detect_industry(self, jobs: List[DocumentJob]) -> tuple[IndustryType, float]:
        """Detect industry type from document content"""
        
        industry_scores = {industry: 0.0 for industry in IndustryType}
        
        for job in jobs:
            content_preview = ""
            try:
                content_preview = job.content.decode('utf-8', errors='ignore')[:1000].lower()
            except:
                pass
            
            filename_lower = job.filename.lower()
            all_text = content_preview + " " + filename_lower
            
            # Score industries based on pattern matches
            for industry, patterns in self.industry_patterns.items():
                for pattern in patterns:
                    if pattern in all_text:
                        industry_scores[industry] += 1.0
                        if job.is_german_compliance:
                            industry_scores[industry] += 0.3
        
        # Find highest scoring industry
        if not any(industry_scores.values()):
            return IndustryType.UNKNOWN, 0.0
        
        best_industry = max(industry_scores.items(), key=lambda x: x[1])
        confidence = min(1.0, best_industry[1] / max(1, len(jobs) * 0.5))
        
        return best_industry[0], confidence
    
    def _detect_german_authority(self, jobs: List[DocumentJob], industry: IndustryType) -> GermanAuthority:
        """Detect relevant German data protection authority"""
        
        authority_scores = {authority: 0.0 for authority in GermanAuthority}
        
        # Industry-based authority mapping
        industry_authority_map = {
            IndustryType.AUTOMOTIVE: GermanAuthority.BAYLDA,  # Bavaria automotive cluster
            IndustryType.HEALTHCARE: GermanAuthority.BFDI,   # Federal health data oversight
            IndustryType.MANUFACTURING: GermanAuthority.LFD_BW,  # Baden-Württemberg engineering
            IndustryType.FINANCIAL: GermanAuthority.BFDI,    # Federal financial oversight
            IndustryType.TECHNOLOGY: GermanAuthority.BFDI    # Federal tech/international
        }
        
        # Give base score for industry match
        if industry in industry_authority_map:
            authority_scores[industry_authority_map[industry]] += 2.0
        
        # Content-based authority detection
        for job in jobs:
            content_preview = ""
            try:
                content_preview = job.content.decode('utf-8', errors='ignore')[:1000].lower()
            except:
                pass
            
            all_text = content_preview + " " + job.filename.lower()
            
            for authority, patterns in self.authority_patterns.items():
                for pattern in patterns:
                    if pattern in all_text:
                        authority_scores[authority] += 1.0
        
        # Return highest scoring authority or unknown
        if not any(authority_scores.values()):
            return GermanAuthority.UNKNOWN
        
        return max(authority_scores.items(), key=lambda x: x[1])[0]
    
    def _analyze_document_types(self, jobs: List[DocumentJob]) -> List[str]:
        """Analyze and categorize document types in portfolio"""
        
        detected_types = set()
        
        for job in jobs:
            # Use compliance indicators from job analysis
            for indicator in job.compliance_indicators:
                detected_types.add(indicator)
            
            # Additional content analysis
            content_preview = ""
            try:
                content_preview = job.content.decode('utf-8', errors='ignore')[:1000].lower()
            except:
                pass
            
            filename_lower = job.filename.lower()
            all_text = content_preview + " " + filename_lower
            
            # Check document type patterns
            for doc_type, patterns in self.document_patterns.items():
                if any(pattern in all_text for pattern in patterns):
                    detected_types.add(doc_type)
        
        return list(detected_types)
    
    def _assess_completeness(self, document_types: List[str], scenario: AuditScenario) -> tuple[float, List[str]]:
        """Assess compliance completeness and identify missing document types"""
        
        # Required documents by scenario
        required_docs = {
            AuditScenario.AUDIT_PREPARATION: [
                'privacy_policy', 'ropa', 'dsfa', 'consent_form',
                'contract', 'training_material'
            ],
            AuditScenario.NEW_SERVICE_LAUNCH: [
                'privacy_policy', 'dsfa', 'consent_form'
            ],
            AuditScenario.INCIDENT_RESPONSE: [
                'breach_response', 'privacy_policy', 'contract'
            ],
            AuditScenario.POLICY_REVIEW: [
                'privacy_policy', 'training_material'
            ],
            AuditScenario.VENDOR_ASSESSMENT: [
                'contract', 'ropa', 'dsfa'
            ]
        }
        
        scenario_requirements = required_docs.get(scenario, ['privacy_policy', 'ropa'])
        
        # Calculate completeness
        found_requirements = sum(1 for req in scenario_requirements if req in document_types)
        completeness = found_requirements / len(scenario_requirements) if scenario_requirements else 1.0
        
        # Identify missing types
        missing_types = [req for req in scenario_requirements if req not in document_types]
        
        return completeness, missing_types
    
    def _generate_smart_actions(
        self,
        scenario: AuditScenario,
        document_types: List[str],
        missing_types: List[str],
        industry: IndustryType
    ) -> List[SmartAction]:
        """Generate smart action suggestions based on context"""
        
        actions = []
        
        # Generate missing document actions
        for missing_type in missing_types[:3]:  # Top 3 missing
            action = self._create_document_generation_action(missing_type, industry)
            if action:
                actions.append(action)
        
        # Scenario-specific actions
        if scenario == AuditScenario.AUDIT_PREPARATION:
            actions.append(SmartAction(
                action_id="export_audit_bundle",
                label="Export Audit-Ready Bundle",
                description="Generate comprehensive audit documentation package",
                priority="high",
                category="export",
                endpoint="/api/v2/export/audit-bundle",
                estimated_time=60
            ))
        
        # Industry-specific actions
        if industry == IndustryType.AUTOMOTIVE:
            actions.append(SmartAction(
                action_id="generate_vehicle_data_policy",
                label="Generate Vehicle Data Policy",
                description="Create automotive-specific data processing policy",
                priority="medium",
                category="generate",
                endpoint="/api/v2/generate/automotive-policy",
                estimated_time=120
            ))
        
        # German compliance actions
        if any('german' in doc_type for doc_type in document_types):
            actions.append(SmartAction(
                action_id="german_authority_check",
                label="German Authority Compliance Check",
                description="Verify compliance with German data protection authorities",
                priority="high",
                category="analyze",
                endpoint="/api/v2/analyze/german-authority",
                estimated_time=90
            ))
        
        # Always offer portfolio analysis
        actions.append(SmartAction(
            action_id="portfolio_gap_analysis",
            label="Complete Portfolio Gap Analysis",
            description="Identify all compliance gaps across document portfolio",
            priority="medium",
            category="analyze",
            endpoint="/api/v2/analyze/portfolio-gaps",
            estimated_time=180
        ))
        
        return actions[:5]  # Limit to top 5 actions
    
    def _create_document_generation_action(self, doc_type: str, industry: IndustryType) -> Optional[SmartAction]:
        """Create document generation action for missing document type"""
        
        action_map = {
            'privacy_policy': {
                'label': 'Generate Privacy Policy',
                'description': 'Create GDPR-compliant privacy policy template',
                'endpoint': '/api/v2/generate/privacy-policy'
            },
            'dsfa': {
                'label': 'Generate DSFA Template',
                'description': 'Create Data Protection Impact Assessment template',
                'endpoint': '/api/v2/generate/dsfa'
            },
            'ropa': {
                'label': 'Generate Records of Processing',
                'description': 'Create Article 30 records of processing template',
                'endpoint': '/api/v2/generate/ropa'
            },
            'consent_form': {
                'label': 'Generate Consent Form',
                'description': 'Create GDPR Article 7 compliant consent template',
                'endpoint': '/api/v2/generate/consent-form'
            }
        }
        
        if doc_type not in action_map:
            return None
        
        action_config = action_map[doc_type]
        
        return SmartAction(
            action_id=f"generate_{doc_type}",
            label=action_config['label'],
            description=action_config['description'],
            priority="high",
            category="generate",
            endpoint=action_config['endpoint'],
            parameters={"industry": industry.value},
            estimated_time=90
        )
    
    def _calculate_portfolio_score(self, jobs: List[DocumentJob], document_types: List[str]) -> float:
        """Calculate overall portfolio compliance score"""
        
        base_score = 0.5
        
        # Bonus for document diversity
        if len(document_types) >= 4:
            base_score += 0.2
        
        # Bonus for German compliance content
        german_ratio = sum(1 for job in jobs if job.is_german_compliance) / len(jobs)
        base_score += german_ratio * 0.2
        
        # Bonus for high-priority documents
        priority_indicators = ['dsfa', 'ropa', 'privacy_policy']
        priority_found = sum(1 for indicator in priority_indicators if indicator in document_types)
        base_score += (priority_found / len(priority_indicators)) * 0.1
        
        return min(1.0, base_score)
    
    def _identify_priority_risks(self, jobs: List[DocumentJob], missing_types: List[str]) -> List[str]:
        """Identify priority compliance risks"""
        
        risks = []
        
        # Missing critical documents
        critical_docs = ['dsfa', 'ropa', 'privacy_policy']
        for doc in critical_docs:
            if doc in missing_types:
                risks.append(f"Missing {doc.replace('_', ' ').title()}")
        
        # German compliance risks
        german_jobs = [job for job in jobs if job.is_german_compliance]
        if german_jobs and len(german_jobs) / len(jobs) > 0.5:
            if 'dsfa' in missing_types:
                risks.append("German DSFA required for high-risk processing")
        
        return risks[:5]  # Top 5 risks
    
    def _identify_quick_wins(self, jobs: List[DocumentJob], document_types: List[str]) -> List[str]:
        """Identify quick compliance wins"""
        
        quick_wins = []
        
        # Document organization
        if len(jobs) > 3:
            quick_wins.append("Organize documents by compliance category")
        
        # German content optimization
        german_count = sum(1 for job in jobs if job.is_german_compliance)
        if german_count > 0:
            quick_wins.append("Optimize German legal terminology")
        
        # Template usage
        if 'privacy_policy' in document_types:
            quick_wins.append("Standardize privacy policy language")
        
        return quick_wins[:3]  # Top 3 quick wins
    
    def _generate_scenario_description(
        self,
        scenario: AuditScenario,
        industry: IndustryType,
        doc_count: int
    ) -> str:
        """Generate human-readable scenario description"""
        
        descriptions = {
            AuditScenario.AUDIT_PREPARATION: f"Detected audit preparation scenario for {industry.value} industry with {doc_count} documents. Focus on comprehensive compliance documentation.",
            AuditScenario.NEW_SERVICE_LAUNCH: f"New service launch detected in {industry.value} sector. Ensure privacy-by-design compliance before go-live.",
            AuditScenario.INCIDENT_RESPONSE: f"Incident response scenario identified. Review breach procedures and notification requirements.",
            AuditScenario.POLICY_REVIEW: f"Policy review and update process detected. Ensure current legal requirements are addressed.",
            AuditScenario.VENDOR_ASSESSMENT: f"Third-party vendor assessment scenario. Focus on data processing agreements and due diligence.",
            AuditScenario.COMPLIANCE_GAP_ANALYSIS: f"Comprehensive compliance gap analysis for {industry.value} organization.",
            AuditScenario.UNKNOWN: f"General compliance analysis for {doc_count} documents in {industry.value} sector."
        }
        
        return descriptions.get(scenario, f"Compliance analysis for {doc_count} documents.")