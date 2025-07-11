# app/services/german_authority_engine/integration/authority_endpoints.py - Fixed Version
from fastapi import UploadFile, HTTPException, Depends
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
from sqlalchemy.ext.asyncio import AsyncSession
import structlog

from app.config import settings  # FIXED: Ensure settings import works
from app.database import get_db_session

logger = structlog.get_logger()

class Big4AuthorityEndpoints:
    """
    Big 4 German Authority Endpoints for Enhanced Compliance Analysis
    
    Provides API endpoints for:
    - Smart authority detection based on business context
    - Authority-specific compliance analysis
    - Multi-authority comparison
    - Industry-specific templates
    """
    
    def __init__(self):
        """Initialize Big 4 Authority Endpoints"""
        logger.info("Initializing Big 4 Authority Endpoints")
        
        # Import detector and analyzer here to avoid circular imports
        try:
            authorities_info = {
                "bfdi": {
                    "id": "bfdi",
                    "name": "Der Bundesbeauftragte für den Datenschutz und die Informationsfreiheit (BfDI)",
                    "english_name": "Federal Commissioner for Data Protection and Freedom of Information",
                    "jurisdiction": "Federal level - cross-border transfers, federal agencies",
                    "location": "Bonn, Germany",
                    "specializations": [
                        "International data transfers",
                        "Cross-border processing",
                        "Federal government compliance",
                        "Large multinational corporations"
                    ],
                    "enforcement_patterns": {
                        "average_fine_amount": "€2.5M",
                        "enforcement_likelihood": "Medium-High",
                        "primary_focus": "Cross-border transfers and large enterprises",
                        "audit_frequency": "Quarterly for large enterprises"
                    },
                    "contact_info": {
                        "website": "https://www.bfdi.bund.de",
                        "email": "poststelle@bfdi.bund.de",
                        "phone": "+49 228 997799-0",
                        "address": "Graurheindorfer Str. 153, 53117 Bonn"
                    },
                    "coverage_percentage": "25%"
                },
                "baylda": {
                    "id": "baylda",
                    "name": "Bayerisches Landesamt für Datenschutzaufsicht (BayLDA)",
                    "english_name": "Bavarian State Office for Data Protection Supervision",
                    "jurisdiction": "Bavaria state - automotive industry focus",
                    "location": "Ansbach, Bavaria",
                    "specializations": [
                        "Automotive industry compliance",
                        "Manufacturing data protection",
                        "Connected vehicle privacy",
                        "Supplier chain data agreements"
                    ],
                    "enforcement_patterns": {
                        "average_fine_amount": "€1.8M",
                        "enforcement_likelihood": "High",
                        "primary_focus": "Automotive and manufacturing sectors",
                        "audit_frequency": "Bi-annual for automotive companies"
                    },
                    "contact_info": {
                        "website": "https://www.lda.bayern.de",
                        "email": "poststelle@lda.bayern.de",
                        "phone": "+49 981 180093-0",
                        "address": "Promenade 18, 91522 Ansbach"
                    },
                    "coverage_percentage": "20%"
                },
                "lfd_bw": {
                    "id": "lfd_bw",
                    "name": "Der Landesbeauftragte für den Datenschutz und die Informationsfreiheit Baden-Württemberg (LfD BW)",
                    "english_name": "State Commissioner for Data Protection and Freedom of Information Baden-Württemberg",
                    "jurisdiction": "Baden-Württemberg state - technology and engineering focus",
                    "location": "Stuttgart, Baden-Württemberg",
                    "specializations": [
                        "Software and technology companies",
                        "Engineering and precision manufacturing",
                        "Research and development data",
                        "Privacy by design implementation"
                    ],
                    "enforcement_patterns": {
                        "average_fine_amount": "€1.2M",
                        "enforcement_likelihood": "Medium",
                        "primary_focus": "Technology and software compliance",
                        "audit_frequency": "Annual for tech companies"
                    },
                    "contact_info": {
                        "website": "https://www.baden-wuerttemberg.datenschutz.de",
                        "email": "poststelle@lfdi.bwl.de",
                        "phone": "+49 711 615541-0",
                        "address": "Lautenschlagerstraße 20, 70173 Stuttgart"
                    },
                    "coverage_percentage": "15%"
                },
                "ldi_nrw": {
                    "id": "ldi_nrw",
                    "name": "Landesbeauftragte für Datenschutz und Informationsfreiheit Nordrhein-Westfalen (LDI NRW)",
                    "english_name": "State Commissioner for Data Protection and Freedom of Information North Rhine-Westphalia",
                    "jurisdiction": "North Rhine-Westphalia state - manufacturing and industry focus",
                    "location": "Düsseldorf, North Rhine-Westphalia",
                    "specializations": [
                        "Heavy manufacturing and industry",
                        "Chemical and pharmaceutical sectors",
                        "Energy and utilities data protection",
                        "Employee data processing"
                    ],
                    "enforcement_patterns": {
                        "average_fine_amount": "€1.5M",
                        "enforcement_likelihood": "Medium-High",
                        "primary_focus": "Manufacturing and industrial compliance",
                        "audit_frequency": "Annual for large manufacturers"
                    },
                    "contact_info": {
                        "website": "https://www.ldi.nrw.de",
                        "email": "poststelle@ldi.nrw.de",
                        "phone": "+49 211 38424-0",
                        "address": "Kavalleriestraße 2-4, 40213 Düsseldorf"
                    },
                    "coverage_percentage": "25%"
                }
            }
            
            # Calculate total coverage
            total_coverage = sum(int(info["coverage_percentage"].rstrip("%")) for info in authorities_info.values())
            
            response = {
                "big4_authorities": authorities_info,
                "summary": {
                    "total_authorities": len(authorities_info),
                    "total_market_coverage": f"{total_coverage}%",
                    "geographic_coverage": [
                        "Federal level (BfDI)",
                        "Bavaria (BayLDA)",
                        "Baden-Württemberg (LfD BW)",
                        "North Rhine-Westphalia (LDI NRW)"
                    ],
                    "industry_specializations": {
                        "automotive": ["baylda"],
                        "technology": ["lfd_bw"],
                        "manufacturing": ["baylda", "ldi_nrw"],
                        "international": ["bfdi"],
                        "federal_agencies": ["bfdi"]
                    }
                },
                "selection_guidance": {
                    "primary_factors": [
                        "Company location and primary operations",
                        "Industry sector and specialization",
                        "Data processing activities and scope",
                        "International transfer requirements"
                    ],
                    "decision_matrix": {
                        "automotive_bavaria": "baylda",
                        "software_baden_wurttemberg": "lfd_bw",
                        "manufacturing_nrw": "ldi_nrw",
                        "international_transfers": "bfdi",
                        "federal_government": "bfdi"
                    }
                },
                "metadata": {
                    "last_updated": datetime.now(timezone.utc).isoformat(),
                    "data_source": "Big 4 Authority Intelligence Engine",
                    "coverage_note": "Covers approximately 85% of German SME market"
                }
            }
            
            return response
            
        except Exception as e:
            logger.error("Failed to retrieve Big 4 authorities info", error=str(e))
            raise HTTPException(
                status_code=500,
                detail=f"Failed to retrieve authorities information: {str(e)}"
            )
    
    # Helper methods
    async def _process_uploaded_files(self, files: List[UploadFile]) -> List[Any]:
        """Process uploaded files into Document objects"""
        
        if len(files) > settings.max_files_per_batch:
            raise HTTPException(
                status_code=400,
                detail=f"Maximum {settings.max_files_per_batch} files allowed per batch"
            )
        
        documents = []
        total_size = 0
        
        for file in files:
            # Validate file type
            if not any(file.filename.lower().endswith(ext) for ext in settings.allowed_extensions):
                raise HTTPException(
                    status_code=400,
                    detail=f"File type not supported: {file.filename}. "
                           f"Allowed: {', '.join(settings.allowed_extensions)}"
                )
            
            # Read content
            content = await file.read()
            file_size = len(content)
            total_size += file_size
            
            # FIXED: Check total size limit using the correct property
            if total_size > settings.max_total_file_size_bytes:
                raise HTTPException(
                    status_code=400,
                    detail=f"Total file size exceeds limit: {settings.max_total_file_size_mb}MB"
                )
            
            # Create Document object
            try:
                decoded_content = content.decode('utf-8', errors='ignore')
            except Exception:
                decoded_content = str(content)
            
            # Create document object (mimicking your existing structure)
            document = type('Document', (), {
                'filename': file.filename,
                'content': decoded_content,
                'file_size': file_size,
                'upload_timestamp': datetime.now(timezone.utc)
            })()
            
            documents.append(document)
        
        return documents
    
    def _generate_key_differences(self, analyses: Dict[str, Dict]) -> List[str]:
        """Generate key differences between authority analyses"""
        
        differences = []
        
        # Compare compliance scores
        scores = {auth: data["compliance_score"] for auth, data in analyses.items()}
        max_score_auth = max(scores, key=scores.get)
        min_score_auth = min(scores, key=scores.get)
        
        differences.append(
            f"Compliance scores vary significantly: {max_score_auth} ({scores[max_score_auth]:.1%}) "
            f"vs {min_score_auth} ({scores[min_score_auth]:.1%})"
        )
        
        # Compare enforcement likelihood
        enforcement = {auth: data["enforcement_likelihood"] for auth, data in analyses.items()}
        max_enforcement_auth = max(enforcement, key=enforcement.get)
        min_enforcement_auth = min(enforcement, key=enforcement.get)
        
        differences.append(
            f"Enforcement risk differs: {max_enforcement_auth} ({enforcement[max_enforcement_auth]:.1%}) "
            f"vs {min_enforcement_auth} ({enforcement[min_enforcement_auth]:.1%})"
        )
        
        # Compare missing requirements
        missing_reqs = {auth: len(data["requirements_missing"]) for auth, data in analyses.items()}
        max_missing_auth = max(missing_reqs, key=missing_reqs.get)
        min_missing_auth = min(missing_reqs, key=missing_reqs.get)
        
        differences.append(
            f"Requirements gap: {max_missing_auth} has {missing_reqs[max_missing_auth]} missing requirements "
            f"vs {min_missing_auth} with {missing_reqs[min_missing_auth]}"
        )
        
        return differences
    
    def _generate_strategic_recommendations(self, analyses: Dict[str, Dict], industry: Optional[str]) -> List[str]:
        """Generate strategic recommendations based on comparison"""
        
        recommendations = []
        
        # Find authority with highest compliance score
        best_compliance = max(analyses.items(), key=lambda x: x[1]["compliance_score"])
        recommendations.append(
            f"Primary recommendation: Choose {best_compliance[0]} for highest compliance alignment"
        )
        
        # Find authority with lowest enforcement risk
        lowest_enforcement = min(analyses.items(), key=lambda x: x[1]["enforcement_likelihood"])
        recommendations.append(
            f"Risk mitigation: {lowest_enforcement[0]} shows lowest enforcement likelihood"
        )
        
        # Industry-specific recommendations
        if industry:
            industry_specific = {
                "automotive": "Consider BayLDA for automotive industry expertise",
                "software": "LfD BW offers specialized software compliance guidance",
                "manufacturing": "LDI NRW has strong manufacturing sector focus",
                "technology": "LfD BW provides technology-focused compliance support"
            }
            
            if industry in industry_specific:
                recommendations.append(industry_specific[industry])
        
        # General strategic advice
        recommendations.extend([
            "Consider establishing relationships with multiple authorities for comprehensive coverage",
            "Implement compliance measures that satisfy the most stringent requirements",
            "Regular compliance audits can improve scores across all jurisdictions"
        ])
        
        return recommendations[:5]  # Limit to top 5 recommendations
    
    def _get_authority_customizations(self, authority: str, industry: str) -> Dict[str, Any]:
        """Get authority-specific template customizations"""
        
        customizations = {
            "bfdi": {
                "focus_areas": ["International data transfers", "Cross-border processing"],
                "additional_documents": ["Transfer Impact Assessment", "Adequacy Decision Review"],
                "specific_clauses": [
                    "International transfer safeguards",
                    "Cross-border processing notifications",
                    "Federal compliance reporting"
                ]
            },
            "baylda": {
                "focus_areas": ["Automotive compliance", "Connected vehicle data"],
                "additional_documents": ["Vehicle Data Processing Notice", "Supplier Data Agreement"],
                "specific_clauses": [
                    "Automotive telematics compliance",
                    "Connected service data protection",
                    "Supplier chain data agreements"
                ]
            },
            "lfd_bw": {
                "focus_areas": ["Privacy by design", "Software compliance"],
                "additional_documents": ["Privacy by Design Documentation", "API Privacy Terms"],
                "specific_clauses": [
                    "Software privacy by design",
                    "API data processing transparency",
                    "Technology compliance frameworks"
                ]
            },
            "ldi_nrw": {
                "focus_areas": ["Manufacturing compliance", "Employee data"],
                "additional_documents": ["Employee Data Policy", "Manufacturing Data Notice"],
                "specific_clauses": [
                    "Manufacturing data protection",
                    "Employee monitoring compliance",
                    "Industrial data processing"
                ]
            }
        }
        
        base_customization = customizations.get(authority, {})
        
        # Add industry-specific elements
        if industry == "automotive" and authority == "baylda":
            base_customization["priority_requirements"] = [
                "Connected vehicle consent mechanisms",
                "Automotive supplier data agreements",
                "Vehicle telematics privacy notices"
            ]
        elif industry == "software" and authority == "lfd_bw":
            base_customization["priority_requirements"] = [
                "Privacy by design implementation",
                "API data processing documentation",
                "User consent management systems"
            ]
        
        return base_customization

# Integration helper functions for enhanced_compliance.py
def create_big4_authority_endpoints() -> Big4AuthorityEndpoints:
    """Factory function to create Big 4 Authority Endpoints instance"""
    return Big4AuthorityEndpoints()

def get_big4_endpoint_routes():
    """
    Get route configuration for adding Big 4 endpoints to enhanced_compliance.py
    
    Returns dictionary with route definitions that can be added to FastAPI router.
    """
    return {
        "analyze_with_authority_detection": {
            "path": "/analyze-with-authority-detection",
            "method": "POST",
            "description": "Smart Authority Detection + Analysis"
        },
        "analyze_authority_specific": {
            "path": "/analyze-authority/{authority_id}",
            "method": "POST", 
            "description": "Authority-Specific Compliance Analysis"
        },
        "compare_authorities": {
            "path": "/compare-authorities",
            "method": "POST",
            "description": "Multi-Authority Compliance Comparison"
        },
        "detect_from_business": {
            "path": "/authorities/detect-from-business",
            "method": "GET",
            "description": "Business Profile Authority Detection"
        },
        "industry_templates": {
            "path": "/templates/industry/{industry}",
            "method": "GET",
            "description": "Industry-Specific Compliance Templates"
        },
        "big4_info": {
            "path": "/authorities/big4",
            "method": "GET",
            "description": "Big 4 German Authorities Information"
        }
    }
            from ..big4.big4_detector import Big4AuthorityDetector
            from ..big4.big4_analyzer import Big4ComplianceAnalyzer
            from .. import get_all_authorities
            
            self.detector = Big4AuthorityDetector()
            self.analyzer = Big4ComplianceAnalyzer()
            self.get_all_authorities = get_all_authorities
            
            logger.info("Big 4 Authority Engine components initialized successfully")
            
        except ImportError as e:
            logger.error(f"Failed to import Big 4 components: {e}")
            self.detector = None
            self.analyzer = None
            self.get_all_authorities = None
    
    async def analyze_with_smart_detection(
        self,
        files: List[UploadFile],
        industry: Optional[str] = None,
        company_location: Optional[str] = None,
        company_size: Optional[str] = None,
        workspace_id: str = None,
        db: AsyncSession = None
    ) -> Dict[str, Any]:
        """
        Smart Authority Detection + Analysis
        
        Automatically detects relevant authorities and provides analysis
        """
        try:
            if not self.detector or not self.analyzer:
                raise HTTPException(
                    status_code=503,
                    detail="Big 4 Authority Engine not available"
                )
            
            # Process uploaded files
            documents = await self._process_uploaded_files(files)
            
            # Detect relevant authorities with business context
            detection_result = await self.detector.detect_relevant_authorities(
                documents=documents,
                suggested_industry=industry,
                suggested_state=company_location
            )
            
            if not detection_result.primary_authority:
                return {
                    "detection_result": "no_authority_detected",
                    "message": "No specific German authority detected for this content",
                    "suggested_actions": [
                        "Review document content for German-specific terms",
                        "Consider using general GDPR compliance analysis"
                    ]
                }
            
            # Perform analysis for detected authority
            analysis = await self.analyzer.analyze_for_authority(
                documents=documents,
                authority=detection_result.primary_authority,
                industry=industry or "unknown"
            )
            
            return {
                "detection_result": "success",
                "detected_authority": {
                    "id": detection_result.primary_authority.value,
                    "name": analysis.authority_name,
                    "confidence": detection_result.detection_confidence,
                    "detection_reasons": detection_result.detection_reasons
                },
                "compliance_analysis": {
                    "compliance_score": analysis.compliance_score,
                    "enforcement_likelihood": analysis.enforcement_likelihood,
                    "penalty_risk_level": analysis.penalty_risk_level,
                    "audit_readiness_score": analysis.audit_readiness_score,
                    "requirements_missing": analysis.requirements_missing,
                    "requirements_met": analysis.requirements_met,
                    "industry_specific_guidance": analysis.industry_specific_guidance,
                    "next_steps": analysis.next_steps,
                    "estimated_penalty_range": analysis.estimated_penalty_range
                },
                "metadata": {
                    "analysis_timestamp": datetime.now(timezone.utc).isoformat(),
                    "documents_analyzed": len(documents),
                    "industry_detected": industry,
                    "company_location": company_location,
                    "workspace_id": workspace_id
                }
            }
            
        except Exception as e:
            logger.error("Smart authority detection failed", error=str(e))
            raise HTTPException(
                status_code=500,
                detail=f"Smart authority detection failed: {str(e)}"
            )
    
    async def analyze_for_specific_authority(
        self,
        authority_id: str,
        files: List[UploadFile],
        industry: Optional[str] = None,
        company_size: Optional[str] = None,
        workspace_id: str = None,
        db: AsyncSession = None
    ) -> Dict[str, Any]:
        """
        Authority-Specific Compliance Analysis
        
        Detailed analysis for a specific German authority
        """
        try:
            if not self.analyzer:
                raise HTTPException(
                    status_code=503,
                    detail="Big 4 Authority Engine not available"
                )
            
            # Validate authority ID
            valid_authorities = ['bfdi', 'baylda', 'lfd_bw', 'ldi_nrw']
            if authority_id not in valid_authorities:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid authority ID: {authority_id}. Valid options: {', '.join(valid_authorities)}"
                )
            
            # Process uploaded files
            documents = await self._process_uploaded_files(files)
            
            # Convert authority_id to enum
            from ..big4.models import GermanAuthority
            authority_enum = GermanAuthority(authority_id)
            
            # Perform analysis
            analysis = await self.analyzer.analyze_for_authority(
                documents=documents,
                authority=authority_enum,
                industry=industry or "unknown"
            )
            
            return {
                "analysis_result": "success",
                "authority": {
                    "id": authority_id,
                    "name": analysis.authority_name
                },
                "compliance_analysis": {
                    "compliance_score": analysis.compliance_score,
                    "enforcement_likelihood": analysis.enforcement_likelihood,
                    "penalty_risk_level": analysis.penalty_risk_level,
                    "audit_readiness_score": analysis.audit_readiness_score,
                    "requirements_missing": analysis.requirements_missing,
                    "requirements_met": analysis.requirements_met,
                    "industry_specific_guidance": analysis.industry_specific_guidance,
                    "enforcement_patterns": analysis.enforcement_patterns,
                    "audit_preparation_steps": analysis.audit_preparation_steps,
                    "estimated_penalty_range": analysis.estimated_penalty_range,
                    "contact_information": analysis.contact_information
                },
                "recommendations": {
                    "immediate_actions": analysis.next_steps[:3],
                    "medium_term_goals": analysis.next_steps[3:6] if len(analysis.next_steps) > 3 else [],
                    "long_term_strategy": analysis.next_steps[6:] if len(analysis.next_steps) > 6 else []
                },
                "metadata": {
                    "analysis_timestamp": datetime.now(timezone.utc).isoformat(),
                    "documents_analyzed": len(documents),
                    "industry": industry,
                    "company_size": company_size,
                    "workspace_id": workspace_id
                }
            }
            
        except Exception as e:
            logger.error("Authority-specific analysis failed", authority_id=authority_id, error=str(e))
            raise HTTPException(
                status_code=500,
                detail=f"Authority-specific analysis failed: {str(e)}"
            )
    
    async def compare_authorities(
        self,
        files: List[UploadFile],
        authorities: List[str],
        industry: Optional[str] = None,
        company_size: Optional[str] = None,
        workspace_id: str = None,
        db: AsyncSession = None
    ) -> Dict[str, Any]:
        """
        Multi-Authority Compliance Comparison
        
        Compare compliance analysis across multiple authorities
        """
        try:
            if not self.analyzer:
                raise HTTPException(
                    status_code=503,
                    detail="Big 4 Authority Engine not available"
                )
            
            # Validate authorities
            valid_authorities = ['bfdi', 'baylda', 'lfd_bw', 'ldi_nrw']
            invalid_authorities = [auth for auth in authorities if auth not in valid_authorities]
            
            if invalid_authorities:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid authority IDs: {', '.join(invalid_authorities)}. Valid options: {', '.join(valid_authorities)}"
                )
            
            if len(authorities) < 2:
                raise HTTPException(
                    status_code=400,
                    detail="At least 2 valid authority IDs required for comparison"
                )
            
            # Process uploaded files
            documents = await self._process_uploaded_files(files)
            
            # Analyze for each authority
            comparison_results = {}
            
            from ..big4.models import GermanAuthority
            
            for authority_id in authorities:
                try:
                    authority_enum = GermanAuthority(authority_id)
                    analysis = await self.analyzer.analyze_for_authority(
                        documents=documents,
                        authority=authority_enum,
                        industry=industry or "unknown"
                    )
                    
                    comparison_results[authority_id] = {
                        "authority_name": analysis.authority_name,
                        "compliance_score": analysis.compliance_score,
                        "enforcement_likelihood": analysis.enforcement_likelihood,
                        "penalty_risk_level": analysis.penalty_risk_level,
                        "audit_readiness_score": analysis.audit_readiness_score,
                        "requirements_missing": analysis.requirements_missing,
                        "estimated_penalty_range": analysis.estimated_penalty_range,
                        "key_advantages": analysis.industry_specific_guidance[:3],
                        "primary_concerns": analysis.requirements_missing[:3]
                    }
                    
                except Exception as e:
                    logger.warning(f"Failed to analyze for authority {authority_id}: {e}")
                    comparison_results[authority_id] = {
                        "error": f"Analysis failed: {str(e)}"
                    }
            
            # Generate comparison insights
            successful_analyses = {k: v for k, v in comparison_results.items() if "error" not in v}
            
            if not successful_analyses:
                raise HTTPException(
                    status_code=500,
                    detail="Failed to analyze for any of the specified authorities"
                )
            
            # Find best and worst options
            best_authority = max(successful_analyses.items(), 
                               key=lambda x: x[1].get("compliance_score", 0))
            
            worst_authority = min(successful_analyses.items(), 
                                key=lambda x: x[1].get("compliance_score", 1))
            
            return {
                "comparison_result": "success",
                "authorities_compared": authorities,
                "detailed_comparison": comparison_results,
                "summary": {
                    "recommended_authority": {
                        "id": best_authority[0],
                        "name": best_authority[1]["authority_name"],
                        "compliance_score": best_authority[1]["compliance_score"],
                        "reasons": [
                            f"Highest compliance score: {best_authority[1]['compliance_score']:.1%}",
                            f"Lower enforcement risk: {best_authority[1]['enforcement_likelihood']:.1%}",
                            "Better alignment with current documentation"
                        ]
                    },
                    "least_favorable": {
                        "id": worst_authority[0],
                        "name": worst_authority[1]["authority_name"],
                        "compliance_score": worst_authority[1]["compliance_score"],
                        "concerns": worst_authority[1]["primary_concerns"]
                    },
                    "key_differences": self._generate_key_differences(successful_analyses),
                    "strategic_recommendations": self._generate_strategic_recommendations(successful_analyses, industry)
                },
                "metadata": {
                    "analysis_timestamp": datetime.now(timezone.utc).isoformat(),
                    "documents_analyzed": len(documents),
                    "industry": industry,
                    "company_size": company_size,
                    "workspace_id": workspace_id
                }
            }
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error("Authority comparison failed", error=str(e))
            raise HTTPException(
                status_code=500,
                detail=f"Authority comparison failed: {str(e)}"
            )
    
    async def detect_relevant_authorities(
        self,
        company_location: str,
        industry: str,
        company_size: str,
        business_activities: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Business Profile Authority Detection
        
        Detect relevant authorities based on business profile without documents
        """
        try:
            if not self.detector:
                raise HTTPException(
                    status_code=503,
                    detail="Big 4 Authority Engine not available"
                )
            
            # Use business profile detection
            detection_result = await self.detector.detect_from_business_profile(
                industry=industry,
                location=company_location,
                company_size=company_size,
                business_activities=business_activities or []
            )
            
            return {
                "detection_result": "success",
                "primary_authority": {
                    "id": detection_result.primary_authority.value,
                    "confidence": detection_result.detection_confidence,
                    "reasons": detection_result.detection_reasons
                },
                "alternative_authorities": [
                    {
                        "id": auth.value,
                        "relevance_score": score
                    }
                    for auth, score in detection_result.alternative_authorities.items()
                ],
                "business_profile": {
                    "industry": industry,
                    "location": company_location,
                    "company_size": company_size,
                    "activities": business_activities
                },
                "next_steps": [
                    f"Consider {detection_result.primary_authority.value} as primary authority",
                    "Upload documents for detailed compliance analysis",
                    "Review authority-specific requirements"
                ]
            }
            
        except Exception as e:
            logger.error("Business profile authority detection failed", error=str(e))
            raise HTTPException(
                status_code=500,
                detail=f"Business profile authority detection failed: {str(e)}"
            )
    
    async def get_industry_template(
        self,
        industry: str,
        authority: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Industry-Specific Compliance Templates
        
        Get pre-configured templates for specific industries
        """
        try:
            # Industry template mapping
            industry_templates = {
                "automotive": {
                    "name": "Automotive Industry GDPR Template",
                    "description": "Connected vehicles, supplier data, customer profiling",
                    "key_documents": [
                        "Privacy Policy for Connected Services",
                        "Supplier Data Processing Agreement",
                        "Customer Data Collection Notice",
                        "Data Sharing Agreement Template",
                        "Incident Response Plan"
                    ],
                    "specific_requirements": [
                        "Vehicle telematics data protection",
                        "Cross-border data transfers for manufacturing",
                        "Customer consent for connected services",
                        "Supplier chain data agreements"
                    ]
                },
                "software": {
                    "name": "Software/SaaS GDPR Template",
                    "description": "Privacy by design, API compliance, user data",
                    "key_documents": [
                        "Privacy by Design Documentation",
                        "API Data Processing Terms",
                        "User Consent Management",
                        "Data Retention Policy",
                        "Security Incident Response"
                    ],
                    "specific_requirements": [
                        "Privacy by design implementation",
                        "API data processing transparency",
                        "User consent management systems",
                        "Data portability mechanisms"
                    ]
                },
                "manufacturing": {
                    "name": "Manufacturing GDPR Template",
                    "description": "IoT compliance, employee monitoring, supply chain",
                    "key_documents": [
                        "Employee Monitoring Policy",
                        "IoT Device Data Policy",
                        "Supply Chain Data Agreement",
                        "Workplace Privacy Notice",
                        "Data Processing Impact Assessment"
                    ],
                    "specific_requirements": [
                        "Employee monitoring compliance",
                        "IoT device data collection rules",
                        "Supply chain data protection",
                        "Workplace surveillance limitations"
                    ]
                },
                "healthcare": {
                    "name": "Healthcare GDPR Template",
                    "description": "Patient data, medical research, health records",
                    "key_documents": [
                        "Patient Data Privacy Notice",
                        "Medical Research Consent",
                        "Health Record Processing Policy",
                        "Data Sharing Agreement (Medical)",
                        "Breach Notification Procedure"
                    ],
                    "specific_requirements": [
                        "Special category data protection",
                        "Medical research consent procedures",
                        "Health record access controls",
                        "Patient rights management"
                    ]
                }
            }
            
            if industry not in industry_templates:
                available_industries = list(industry_templates.keys())
                raise HTTPException(
                    status_code=400,
                    detail=f"Industry template not available for '{industry}'. Available: {', '.join(available_industries)}"
                )
            
            template = industry_templates[industry].copy()
            
            # Add authority-specific customizations if specified
            if authority:
                template["authority_customizations"] = self._get_authority_customizations(authority, industry)
            
            template["generation_timestamp"] = datetime.now(timezone.utc).isoformat()
            template["industry"] = industry
            template["authority"] = authority
            
            return {
                "template_result": "success",
                "industry_template": template,
                "usage_instructions": [
                    "Review and customize the template documents for your specific use case",
                    "Ensure all industry-specific requirements are addressed",
                    "Consider authority-specific customizations if applicable",
                    "Test the compliance framework with sample data"
                ]
            }
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error("Industry template generation failed", industry=industry, error=str(e))
            raise HTTPException(
                status_code=500,
                detail=f"Industry template generation failed: {str(e)}"
            )
    
    async def get_all_big4_authorities_info(self) -> Dict[str, Any]:
        """
        Big 4 German Authorities Information
        
        Complete information about all Big 4 authorities
        """
        try: