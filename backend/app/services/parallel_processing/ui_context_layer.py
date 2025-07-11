# app/services/parallel_processing/ui_context_layer.py - FIXED: Complete Implementation
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
import structlog
import re

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
    LDI_NRW = "ldi_nrw"  # North Rhine-Westphalia
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
        # IMPROVED: Better import handling with fallback
        self.DocumentJob = None
        self._initialize_document_job()
        
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
                'breach', 'incident', 'security event', 'datenpanne',
                'data leak', 'unauthorized access', 'cyberattack'
            ],
            'policy_review': [
                'policy review', 'richtlinien überprüfung',
                'policy update', 'revision', 'überarbeitung'
            ],
            'vendor_assessment': [
                'vendor', 'supplier', 'third party', 'lieferant',
                'service provider', 'contractor', 'partner'
            ]
        }
        
        # Scenario detection patterns
        self.scenario_patterns = {
            AuditScenario.AUDIT_PREPARATION: [
                'audit', 'prüfung', 'inspection', 'compliance check',
                'authority visit', 'behördliche prüfung', 'kontrolle'
            ],
            AuditScenario.NEW_SERVICE_LAUNCH: [
                'new service', 'launch', 'product release', 'rollout',
                'market entry', 'service introduction', 'neuer service'
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
        
        # Industry detection patterns
        self.industry_patterns = {
            IndustryType.AUTOMOTIVE: [
                'automotive', 'car', 'vehicle', 'bmw', 'mercedes', 'audi',
                'porsche', 'manufacturing', 'fahrzeug', 'automobil'
            ],
            IndustryType.HEALTHCARE: [
                'healthcare', 'medical', 'hospital', 'patient', 'gesundheit',
                'medizin', 'klinik', 'arzt', 'health data'
            ],
            IndustryType.MANUFACTURING: [
                'manufacturing', 'production', 'factory', 'industrial',
                'herstellung', 'produktion', 'fertigung', 'industrie'
            ],
            IndustryType.FINANCIAL: [
                'financial', 'bank', 'insurance', 'payment', 'fintech',
                'finanzen', 'versicherung', 'zahlung', 'kredit'
            ],
            IndustryType.TECHNOLOGY: [
                'technology', 'software', 'app', 'digital', 'tech',
                'technologie', 'software', 'digital', 'it'
            ],
            IndustryType.RETAIL: [
                'retail', 'shop', 'store', 'e-commerce', 'customer',
                'handel', 'laden', 'geschäft', 'verkauf'
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
            GermanAuthority.LDI_NRW: [
                'ldi', 'nrw', 'north rhine', 'düsseldorf', 'cologne',
                'manufacturing', 'industry'
            ],
            GermanAuthority.BERLIN_BFDI: [
                'berlin', 'hauptstadt', 'government', 'regierung'
            ]
        }
    
    def _initialize_document_job(self):
        """IMPROVED: Initialize DocumentJob with better error handling"""
        try:
            from .job_queue import DocumentJob
            self.DocumentJob = DocumentJob
            logger.debug("DocumentJob successfully imported")
        except ImportError as e:
            logger.warning(f"DocumentJob not available: {e} - Some features may be limited")
            self.DocumentJob = None
        except Exception as e:
            logger.error(f"Unexpected error importing DocumentJob: {e}")
            self.DocumentJob = None
    
    def detect_german_content(self, content) -> bool:
        """
        Public method for German content detection - can handle bytes or str
        
        Args:
            content: Document content (bytes, str, or other)
            
        Returns:
            bool: True if German content detected
        """
        try:
            # Handle different content types
            if isinstance(content, bytes):
                content_str = content.decode('utf-8', errors='ignore')
            elif isinstance(content, str):
                content_str = content
            else:
                content_str = str(content)
            
            return self._detect_german_content_internal(content_str)
            
        except Exception as e:
            logger.warning(f"German content detection failed: {e}")
            return False
    
    def _detect_german_content_internal(self, content: str) -> bool:
        """
        Internal German content detection logic
        
        Args:
            content: String content to analyze
            
        Returns:
            bool: True if German content detected
        """
        german_indicators = [
            "dsgvo", "datenschutz", "verarbeitung", "einwilligung",
            "personenbezogene daten", "betroffenenrechte", "dsfa",
            "aufsichtsbehörde", "rechtsgrundlage", "artikel",
            "datenschutzbeauftragte", "datenschutzbeauftragter",
            "auftragsverarbeitung", "gemeinsame verantwortlichkeit",
            "datenschutz-folgenabschätzung", "verfahrensverzeichnis"
        ]
        
        content_lower = content.lower()
        german_count = sum(1 for indicator in german_indicators if indicator in content_lower)
        
        # Also check for German legal article references
        article_patterns = [
            r"art\.\s*\d+", r"artikel\s*\d+", r"abs\.\s*\d+",
            r"§\s*\d+", r"bdsg", r"tmg"
        ]
        
        article_matches = sum(1 for pattern in article_patterns 
                             if re.search(pattern, content_lower))
        
        logger.debug(f"German content detection: {german_count} terms, {article_matches} articles")
        
        return german_count >= 2 or article_matches >= 1
    
    # Legacy method for backward compatibility
    def _detect_german_content(self, content) -> bool:
        """Legacy method name - redirects to public method for backward compatibility"""
        return self.detect_german_content(content)
    
    def analyze_ui_context(self, jobs) -> UIContext:
        """
        Analyze document jobs to create intelligent UI context
        
        Args:
            jobs: List of DocumentJob objects or similar structures
            
        Returns:
            UIContext: Comprehensive context for UI automation
        """
        
        logger.info(
            "Analyzing UI context",
            job_count=len(jobs),
            german_jobs=sum(1 for job in jobs if self._is_german_job(job))
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
        german_percentage = sum(1 for job in jobs if self._is_german_job(job)) / len(jobs) * 100 if jobs else 0
        
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
    
    def _is_german_job(self, job) -> bool:
        """Check if job is German compliance related - flexible for different job types"""
        try:
            # Try DocumentJob interface first
            if hasattr(job, 'is_german_compliance'):
                return job.is_german_compliance
            
            # Try content-based detection
            if hasattr(job, 'content'):
                return self.detect_german_content(job.content)
            
            # Try filename-based detection
            if hasattr(job, 'filename'):
                return self._detect_german_from_filename(job.filename)
            
            return False
            
        except Exception:
            return False
    
    def _detect_german_from_filename(self, filename: str) -> bool:
        """Detect German content from filename"""
        german_filename_indicators = [
            'datenschutz', 'dsgvo', 'verfahrensverzeichnis', 'dsfa',
            'sicherheit', 'richtlinie', 'verarbeitung', 'auftragsverarbeitung'
        ]
        
        filename_lower = filename.lower()
        return any(indicator in filename_lower for indicator in german_filename_indicators)
    
    def _detect_scenario(self, jobs) -> Tuple[AuditScenario, float]:
        """Detect the most likely compliance scenario"""
        
        scenario_scores = {scenario: 0.0 for scenario in AuditScenario}
        
        for job in jobs:
            content_preview = self._get_job_content_preview(job)
            filename_lower = self._get_job_filename(job).lower()
            all_text = content_preview + " " + filename_lower
            
            # Score each scenario based on pattern matches
            for scenario, patterns in self.scenario_patterns.items():
                for pattern in patterns:
                    if pattern in all_text:
                        scenario_scores[scenario] += 1.0
                        if self._is_german_job(job):
                            scenario_scores[scenario] += 0.5  # Bonus for German docs
        
        # Special scenario detection logic
        doc_types = self._get_all_compliance_indicators(jobs)
        
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
        confidence = min(1.0, best_scenario[1] / len(jobs)) if jobs else 0.0
        
        return best_scenario[0], confidence
    
    def _detect_industry(self, jobs) -> Tuple[IndustryType, float]:
        """Detect industry type from document content"""
        
        industry_scores = {industry: 0.0 for industry in IndustryType}
        
        for job in jobs:
            content_preview = self._get_job_content_preview(job)
            filename_lower = self._get_job_filename(job).lower()
            all_text = content_preview + " " + filename_lower
            
            # Score industries based on pattern matches
            for industry, patterns in self.industry_patterns.items():
                for pattern in patterns:
                    if pattern in all_text:
                        industry_scores[industry] += 1.0
                        if self._is_german_job(job):
                            industry_scores[industry] += 0.3
        
        # Find highest scoring industry
        if not any(industry_scores.values()):
            return IndustryType.UNKNOWN, 0.0
        
        best_industry = max(industry_scores.items(), key=lambda x: x[1])
        confidence = min(1.0, best_industry[1] / max(1, len(jobs) * 0.5)) if jobs else 0.0
        
        return best_industry[0], confidence
    
    def _detect_german_authority(self, jobs, industry: IndustryType) -> GermanAuthority:
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
            content_preview = self._get_job_content_preview(job)
            filename_lower = self._get_job_filename(job).lower()
            all_text = content_preview + " " + filename_lower
            
            for authority, patterns in self.authority_patterns.items():
                for pattern in patterns:
                    if pattern in all_text:
                        authority_scores[authority] += 1.0
        
        # Find highest scoring authority
        if not any(authority_scores.values()):
            return GermanAuthority.UNKNOWN
        
        best_authority = max(authority_scores.items(), key=lambda x: x[1])
        return best_authority[0]
    
    def _get_job_content_preview(self, job) -> str:
        """Get content preview from job - flexible for different job types"""
        try:
            if hasattr(job, 'content'):
                content = job.content
                if isinstance(content, bytes):
                    return content.decode('utf-8', errors='ignore')[:1000].lower()
                elif isinstance(content, str):
                    return content[:1000].lower()
            return ""
        except Exception:
            return ""
    
    def _get_job_filename(self, job) -> str:
        """Get filename from job - flexible for different job types"""
        try:
            if hasattr(job, 'filename'):
                return job.filename
            return ""
        except Exception:
            return ""
    
    def _get_all_compliance_indicators(self, jobs) -> List[str]:
        """Get all compliance indicators from jobs"""
        indicators = []
        for job in jobs:
            if hasattr(job, 'compliance_indicators'):
                indicators.extend(job.compliance_indicators)
            else:
                # Fallback: detect from filename
                filename = self._get_job_filename(job)
                for doc_type, patterns in self.document_patterns.items():
                    if any(pattern in filename.lower() for pattern in patterns):
                        indicators.append(doc_type)
        return indicators
    
    def _analyze_document_types(self, jobs) -> List[str]:
        """Analyze what document types are present"""
        return self._get_all_compliance_indicators(jobs)
    
    def _assess_completeness(self, document_types: List[str], scenario: AuditScenario) -> Tuple[float, List[str]]:
        """Assess compliance completeness and identify missing documents"""
        
        # FIXED: Complete the required_docs dictionary properly
        required_docs = {
            AuditScenario.AUDIT_PREPARATION: ['privacy_policy', 'ropa', 'dsfa', 'contract'],
            AuditScenario.NEW_SERVICE_LAUNCH: ['privacy_policy', 'dsfa', 'consent_form'],
            AuditScenario.INCIDENT_RESPONSE: ['breach_response', 'privacy_policy'],
            AuditScenario.POLICY_REVIEW: ['privacy_policy', 'contract'],
            AuditScenario.VENDOR_ASSESSMENT: ['contract', 'vendor_assessment'],
            AuditScenario.COMPLIANCE_GAP_ANALYSIS: ['privacy_policy', 'ropa', 'dsfa'],
            AuditScenario.UNKNOWN: []
        }
        
        required_for_scenario = required_docs.get(scenario, [])
        present_types = set(document_types)
        required_types = set(required_for_scenario)
        
        if not required_types:
            return 0.8, []  # Default completeness for unknown scenarios
        
        missing_types = list(required_types - present_types)
        completeness = len(present_types & required_types) / len(required_types)
        
        return completeness, missing_types
    
    def _generate_smart_actions(self, scenario: AuditScenario, document_types: List[str], 
                               missing_types: List[str], industry: IndustryType) -> List[SmartAction]:
        """Generate smart action suggestions based on context"""
        
        actions = []
        
        # Missing document actions
        for missing_type in missing_types:
            action = SmartAction(
                action_id=f"generate_{missing_type}",
                label=f"Generate {missing_type.replace('_', ' ').title()}",
                description=f"Create a {missing_type.replace('_', ' ')} template for {industry.value} industry",
                priority="high",
                category="generate",
                endpoint=f"/api/v2/templates/{missing_type}",
                parameters={"industry": industry.value},
                estimated_time=60
            )
            actions.append(action)
        
        # Scenario-specific actions
        if scenario == AuditScenario.AUDIT_PREPARATION:
            actions.append(SmartAction(
                action_id="export_audit_report",
                label="Export Audit Report",
                description="Generate comprehensive audit preparation report",
                priority="high",
                category="export",
                endpoint="/api/v2/reports/audit",
                estimated_time=30
            ))
        
        elif scenario == AuditScenario.INCIDENT_RESPONSE:
            actions.append(SmartAction(
                action_id="breach_notification_template",
                label="Generate Breach Notification",
                description="Create data breach notification templates",
                priority="high",
                category="generate",
                endpoint="/api/v2/templates/breach_notification",
                estimated_time=45
            ))
        
        # Industry-specific actions
        if industry == IndustryType.AUTOMOTIVE:
            actions.append(SmartAction(
                action_id="automotive_compliance_check",
                label="Automotive Compliance Check",
                description="Specialized automotive industry compliance analysis",
                priority="medium",
                category="analyze",
                endpoint="/api/v2/industry/automotive/analyze",
                estimated_time=90
            ))
        
        return actions[:5]  # Limit to top 5 actions
    
    def _calculate_portfolio_score(self, jobs, document_types: List[str]) -> float:
        """Calculate overall portfolio compliance score"""
        
        base_score = 0.3  # Base score for having documents
        
        # Bonus for document variety
        variety_bonus = min(0.3, len(set(document_types)) * 0.05)
        
        # Bonus for German compliance
        german_jobs = sum(1 for job in jobs if self._is_german_job(job))
        german_bonus = min(0.2, (german_jobs / len(jobs)) * 0.2) if jobs else 0
        
        # Bonus for critical documents
        critical_docs = ['privacy_policy', 'ropa', 'dsfa']
        critical_bonus = sum(0.05 for doc in critical_docs if doc in document_types)
        
        return min(1.0, base_score + variety_bonus + german_bonus + critical_bonus)
    
    def _identify_priority_risks(self, jobs, missing_types: List[str]) -> List[str]:
        """Identify priority compliance risks"""
        
        risks = []
        
        # Missing critical documents
        critical_missing = {
            'privacy_policy': "Missing Privacy Policy - Required for GDPR compliance",
            'ropa': "Missing Records of Processing Activities - Article 30 GDPR violation risk",
            'dsfa': "Missing Data Protection Impact Assessment - High-risk processing detected"
        }
        
        for missing in missing_types:
            if missing in critical_missing:
                risks.append(critical_missing[missing])
        
        # Content-based risks
        german_jobs = sum(1 for job in jobs if self._is_german_job(job))
        if german_jobs > 0 and 'ropa' in missing_types:
            risks.append("German content detected without proper ROPA documentation")
        
        return risks[:3]  # Top 3 risks
    
    def _identify_quick_wins(self, jobs, document_types: List[str]) -> List[str]:
        """Identify quick compliance improvements"""
        
        wins = []
        
        if 'privacy_policy' in document_types:
            wins.append("Privacy Policy detected - good foundation for compliance")
        
        if any(self._is_german_job(job) for job in jobs):
            wins.append("German compliance content detected - leverage for EU market")
        
        if len(document_types) >= 3:
            wins.append("Good document variety - comprehensive compliance approach")
        
        wins.append("Export compliance report for stakeholder review")
        wins.append("Generate missing document templates to close gaps")
        
        return wins[:3]  # Top 3 quick wins
    
    def _generate_scenario_description(self, scenario: AuditScenario, 
                                     industry: IndustryType, doc_count: int) -> str:
        """Generate human-readable scenario description"""
        
        scenario_descriptions = {
            AuditScenario.AUDIT_PREPARATION: f"Audit preparation detected for {industry.value} industry with {doc_count} documents",
            AuditScenario.NEW_SERVICE_LAUNCH: f"New service launch compliance check for {industry.value} sector",
            AuditScenario.INCIDENT_RESPONSE: f"Data incident response preparation in {industry.value} context",
            AuditScenario.POLICY_REVIEW: f"Policy review and update process for {industry.value} organization",
            AuditScenario.VENDOR_ASSESSMENT: f"Third-party vendor compliance assessment in {industry.value}",
            AuditScenario.COMPLIANCE_GAP_ANALYSIS: f"Compliance gap analysis for {industry.value} operations",
            AuditScenario.UNKNOWN: f"General compliance analysis for {industry.value} with {doc_count} documents"
        }
        
        return scenario_descriptions.get(scenario, f"Compliance analysis for {industry.value} industry")