# backend/app/services/german_authority_engine/integration/authority_endpoints.py
"""
Big 4 Authority API Integration Endpoints

This module provides the integration layer between enhanced_compliance.py
and the Big 4 German Authority Engine. It adds new endpoints for:

- Smart authority detection and analysis
- Multi-authority comparison
- Industry-specific compliance analysis
- Authority-specific recommendations

These endpoints extend enhanced_compliance.py without modifying existing functionality.
"""

from fastapi import HTTPException, UploadFile, File, Form, Query, Depends
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
from sqlalchemy.ext.asyncio import AsyncSession
import structlog

from app.models.database import Document
from app.database import get_db_session
from app.config import settings

# Big 4 Authority Engine imports
from ..big4.big4_profiles import Big4Authority, get_big4_authority_profile, get_all_big4_authorities
from ..big4.big4_detector import Big4AuthorityDetector
from ..big4.big4_analyzer import Big4ComplianceAnalyzer
from ..big4.big4_templates import Big4IndustryTemplateEngine, IndustryTemplate
from .multi_authority_analyzer import MultiAuthorityAnalyzer
from .response_formatter import Big4ResponseFormatter

logger = structlog.get_logger()

class Big4AuthorityEndpoints:
    """
    Big 4 Authority API endpoints for integration with enhanced_compliance.py
    
    This class provides methods that can be added as FastAPI endpoints
    to extend the existing compliance analysis functionality.
    """
    
    def __init__(self):
        self.detector = Big4AuthorityDetector()
        self.analyzer = Big4ComplianceAnalyzer()
        self.template_engine = Big4IndustryTemplateEngine()
        self.multi_analyzer = MultiAuthorityAnalyzer()
        self.formatter = Big4ResponseFormatter()
        
        logger.info("Big 4 Authority Endpoints initialized")
    
    async def analyze_with_smart_detection(
        self,
        files: List[UploadFile],
        industry: Optional[str] = None,
        company_location: Optional[str] = None,
        company_size: Optional[str] = None,
        workspace_id: str = "demo",
        db: AsyncSession = None
    ) -> Dict[str, Any]:
        """
        Smart authority detection and analysis endpoint
        
        Automatically detects relevant Big 4 authorities and provides
        comprehensive analysis with multi-authority comparison.
        
        Usage in enhanced_compliance.py:
        @router.post("/analyze-with-authority-detection")
        async def analyze_with_authority_detection(...):
            endpoints = Big4AuthorityEndpoints()
            return await endpoints.analyze_with_smart_detection(...)
        """
        
        start_time = datetime.now(timezone.utc)
        
        logger.info(
            "Starting smart authority detection analysis",
            files=len(files),
            industry=industry,
            location=company_location,
            company_size=company_size
        )
        
        try:
            # Validate and process files
            documents = await self._process_uploaded_files(files)
            
            # Perform smart detection and analysis
            analysis_result = await self.analyzer.analyze_with_smart_detection(
                documents=documents,
                industry=industry,
                company_location=company_location,
                company_size=company_size
            )
            
            # Get industry template guidance if industry provided
            template_guidance = None
            if industry:
                try:
                    industry_enum = IndustryTemplate(industry.lower())
                    template_guidance = self.template_engine.get_industry_requirements_checklist(
                        industry_enum, analysis_result.primary_analysis.authority_id
                    )
                except ValueError:
                    logger.warning(f"Unknown industry for template: {industry}")
            
            # Format comprehensive response
            response = await self.formatter.format_smart_detection_response(
                analysis_result=analysis_result,
                template_guidance=template_guidance,
                processing_time=(datetime.now(timezone.utc) - start_time).total_seconds(),
                workspace_id=workspace_id
            )
            
            logger.info(
                "Smart authority detection completed",
                primary_authority=analysis_result.primary_analysis.authority_id,
                processing_time=response["processing_metadata"]["processing_time_seconds"]
            )
            
            return response
            
        except Exception as e:
            logger.error(
                "Smart authority detection failed",
                error=str(e),
                industry=industry,
                location=company_location
            )
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
        workspace_id: str = "demo",
        db: AsyncSession = None
    ) -> Dict[str, Any]:
        """
        Authority-specific analysis endpoint
        
        Provides detailed analysis for a specific Big 4 authority
        with industry-specific guidance and recommendations.
        
        Usage in enhanced_compliance.py:
        @router.post("/analyze-authority/{authority_id}")
        async def analyze_authority_specific(...):
            endpoints = Big4AuthorityEndpoints()
            return await endpoints.analyze_for_specific_authority(...)
        """
        
        start_time = datetime.now(timezone.utc)
        
        # Validate authority
        try:
            authority = Big4Authority(authority_id.lower())
        except ValueError:
            valid_authorities = [auth.value for auth in Big4Authority]
            raise HTTPException(
                status_code=400,
                detail=f"Invalid authority ID: {authority_id}. Valid options: {valid_authorities}"
            )
        
        logger.info(
            "Starting authority-specific analysis",
            authority=authority.value,
            files=len(files),
            industry=industry,
            company_size=company_size
        )
        
        try:
            # Process files
            documents = await self._process_uploaded_files(files)
            
            # Perform authority-specific analysis
            analysis = await self.analyzer.analyze_for_authority(
                documents=documents,
                authority=authority,
                industry=industry,
                company_size=company_size
            )
            
            # Get industry template guidance
            template_guidance = None
            if industry:
                try:
                    industry_enum = IndustryTemplate(industry.lower())
                    template_guidance = self.template_engine.get_industry_requirements_checklist(
                        industry_enum, authority
                    )
                except ValueError:
                    logger.warning(f"Unknown industry for template: {industry}")
            
            # Format response
            response = await self.formatter.format_authority_analysis_response(
                analysis=analysis,
                template_guidance=template_guidance,
                processing_time=(datetime.now(timezone.utc) - start_time).total_seconds(),
                workspace_id=workspace_id
            )
            
            logger.info(
                "Authority-specific analysis completed",
                authority=authority.value,
                compliance_score=analysis.compliance_score,
                processing_time=response["processing_metadata"]["processing_time_seconds"]
            )
            
            return response
            
        except Exception as e:
            logger.error(
                "Authority-specific analysis failed",
                authority=authority.value,
                error=str(e)
            )
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
        workspace_id: str = "demo",
        db: AsyncSession = None
    ) -> Dict[str, Any]:
        """
        Multi-authority comparison endpoint
        
        Compares compliance analysis across multiple Big 4 authorities
        to help determine optimal jurisdiction strategy.
        
        Usage in enhanced_compliance.py:
        @router.post("/compare-authorities")
        async def compare_authority_compliance(...):
            endpoints = Big4AuthorityEndpoints()
            return await endpoints.compare_authorities(...)
        """
        
        start_time = datetime.now(timezone.utc)
        
        # Validate authorities
        validated_authorities = []
        for auth_id in authorities:
            try:
                authority = Big4Authority(auth_id.lower())
                validated_authorities.append(authority)
            except ValueError:
                logger.warning(f"Invalid authority ID ignored: {auth_id}")
        
        if len(validated_authorities) < 2:
            raise HTTPException(
                status_code=400,
                detail="At least 2 valid authority IDs required for comparison"
            )
        
        logger.info(
            "Starting multi-authority comparison",
            authorities=[auth.value for auth in validated_authorities],
            files=len(files),
            industry=industry
        )
        
        try:
            # Process files
            documents = await self._process_uploaded_files(files)
            
            # Perform multi-authority analysis
            comparison_result = await self.multi_analyzer.compare_authorities(
                documents=documents,
                authorities=validated_authorities,
                industry=industry,
                company_size=company_size
            )
            
            # Format response
            response = await self.formatter.format_comparison_response(
                comparison_result=comparison_result,
                processing_time=(datetime.now(timezone.utc) - start_time).total_seconds(),
                workspace_id=workspace_id
            )
            
            logger.info(
                "Multi-authority comparison completed",
                authorities=[auth.value for auth in validated_authorities],
                recommended_primary=comparison_result.recommended_primary_authority,
                processing_time=response["processing_metadata"]["processing_time_seconds"]
            )
            
            return response
            
        except Exception as e:
            logger.error(
                "Multi-authority comparison failed",
                authorities=[auth.value for auth in validated_authorities],
                error=str(e)
            )
            raise HTTPException(
                status_code=500,
                detail=f"Multi-authority comparison failed: {str(e)}"
            )
    
    async def detect_relevant_authorities(
        self,
        company_location: str,
        industry: str,
        company_size: str,
        business_activities: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Authority detection endpoint (no documents required)
        
        Helps users identify relevant German authorities based on
        business profile before document analysis.
        
        Usage in enhanced_compliance.py:
        @router.get("/authorities/detect-from-business")
        async def detect_authorities_from_business(...):
            endpoints = Big4AuthorityEndpoints()
            return await endpoints.detect_relevant_authorities(...)
        """
        
        logger.info(
            "Starting business-based authority detection",
            location=company_location,
            industry=industry,
            company_size=company_size
        )
        
        try:
            # Perform business-based detection
            detection_result = await self.detector.suggest_authorities_for_business(
                location=company_location,
                industry=industry,
                company_size=company_size,
                business_activities=business_activities
            )
            
            # Get recommended template
            template_recommendation = None
            try:
                industry_enum = IndustryTemplate(industry.lower())
                template_recommendation = self.template_engine.get_template(industry_enum)
            except ValueError:
                logger.warning(f"No template available for industry: {industry}")
            
            # Format response
            response = await self.formatter.format_detection_response(
                detection_result=detection_result,
                template_recommendation=template_recommendation
            )
            
            logger.info(
                "Business-based authority detection completed",
                primary_authority=detection_result.primary_authority.value,
                confidence=detection_result.detection_confidence
            )
            
            return response
            
        except Exception as e:
            logger.error(
                "Business-based authority detection failed",
                location=company_location,
                industry=industry,
                error=str(e)
            )
            raise HTTPException(
                status_code=500,
                detail=f"Authority detection failed: {str(e)}"
            )
    
    async def get_industry_template(
        self,
        industry: str,
        authority: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Industry template endpoint
        
        Returns compliance template and checklist for specific industry
        and optionally specific authority.
        
        Usage in enhanced_compliance.py:
        @router.get("/templates/industry/{industry}")
        async def get_industry_compliance_template(...):
            endpoints = Big4AuthorityEndpoints()
            return await endpoints.get_industry_template(...)
        """
        
        try:
            industry_enum = IndustryTemplate(industry.lower())
        except ValueError:
            available_industries = [template.value for template in IndustryTemplate]
            raise HTTPException(
                status_code=400,
                detail=f"Invalid industry: {industry}. Available: {available_industries}"
            )
        
        authority_enum = None
        if authority:
            try:
                authority_enum = Big4Authority(authority.lower())
            except ValueError:
                valid_authorities = [auth.value for auth in Big4Authority]
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid authority: {authority}. Valid options: {valid_authorities}"
                )
        
        logger.info(
            "Getting industry template",
            industry=industry,
            authority=authority
        )
        
        try:
            # Get template
            template = self.template_engine.get_template(industry_enum)
            if not template:
                raise HTTPException(
                    status_code=404,
                    detail=f"No template available for industry: {industry}"
                )
            
            # Get authority-specific checklist if authority provided
            checklist = None
            if authority_enum:
                checklist = self.template_engine.get_industry_requirements_checklist(
                    industry_enum, authority_enum
                )
            
            # Format response
            response = await self.formatter.format_template_response(
                template=template,
                checklist=checklist,
                authority=authority_enum
            )
            
            return response
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(
                "Industry template retrieval failed",
                industry=industry,
                authority=authority,
                error=str(e)
            )
            raise HTTPException(
                status_code=500,
                detail=f"Template retrieval failed: {str(e)}"
            )
    
    async def get_all_big4_authorities_info(self) -> Dict[str, Any]:
        """
        Big 4 authorities information endpoint
        
        Returns information about all Big 4 German authorities
        including their profiles and focus areas.
        
        Usage in enhanced_compliance.py:
        @router.get("/authorities/big4")
        async def get_big4_authorities(...):
            endpoints = Big4AuthorityEndpoints()
            return await endpoints.get_all_big4_authorities_info()
        """
        
        try:
            all_authorities = get_all_big4_authorities()
            
            response = {
                "authorities": [],
                "coverage_info": {
                    "total_authorities": len(all_authorities),
                    "german_market_coverage": "70%",
                    "business_coverage": sum(
                        profile.covers_businesses for profile in all_authorities.values()
                    )
                }
            }
            
            for authority, profile in all_authorities.items():
                authority_info = {
                    "authority_id": authority.value,
                    "name": profile.name,
                    "name_english": profile.name_english,
                    "jurisdiction": profile.jurisdiction,
                    "state_code": profile.state_code,
                    
                    "contact": {
                        "website": profile.website,
                        "email": profile.email,
                        "phone": profile.phone
                    },
                    
                    "characteristics": {
                        "enforcement_style": profile.enforcement_profile.enforcement_style,
                        "audit_frequency": profile.enforcement_profile.audit_frequency,
                        "avg_penalty": profile.enforcement_profile.avg_penalty_amount,
                        "sme_focus": profile.sme_focus,
                        "market_share": profile.market_share
                    },
                    
                    "specializations": {
                        "industries": profile.industry_specializations,
                        "priority_articles": profile.priority_articles,
                        "unique_requirements": profile.unique_requirements
                    }
                }
                
                response["authorities"].append(authority_info)
            
            return response
            
        except Exception as e:
            logger.error("Failed to retrieve Big 4 authorities info", error=str(e))
            raise HTTPException(
                status_code=500,
                detail=f"Failed to retrieve authorities information: {str(e)}"
            )
    
    # Helper methods
    async def _process_uploaded_files(self, files: List[UploadFile]) -> List[Document]:
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
            
            # Check total size limit
            if total_size > settings.max_total_file_size:
                raise HTTPException(
                    status_code=400,
                    detail=f"Total file size exceeds limit: {settings.max_total_file_size} bytes"
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
            "description": "Smart authority detection and analysis",
            "parameters": [
                "files: List[UploadFile]",
                "industry: Optional[str] = Form(None)",
                "company_location: Optional[str] = Form(None)",
                "company_size: Optional[str] = Form(None)",
                "workspace_id: str = Form(default=DEMO_WORKSPACE_ID)"
            ]
        },
        
        "analyze_authority_specific": {
            "path": "/analyze-authority/{authority_id}",
            "method": "POST", 
            "description": "Authority-specific compliance analysis",
            "parameters": [
                "authority_id: str",
                "files: List[UploadFile]",
                "industry: Optional[str] = Form(None)",
                "company_size: Optional[str] = Form(None)",
                "workspace_id: str = Form(default=DEMO_WORKSPACE_ID)"
            ]
        },
        
        "compare_authorities": {
            "path": "/compare-authorities",
            "method": "POST",
            "description": "Multi-authority compliance comparison",
            "parameters": [
                "files: List[UploadFile]",
                "authorities: List[str] = Form(...)",
                "industry: Optional[str] = Form(None)",
                "company_size: Optional[str] = Form(None)",
                "workspace_id: str = Form(default=DEMO_WORKSPACE_ID)"
            ]
        },
        
        "detect_authorities_from_business": {
            "path": "/authorities/detect-from-business",
            "method": "GET",
            "description": "Detect relevant authorities from business profile",
            "parameters": [
                "company_location: str = Query(...)",
                "industry: str = Query(...)", 
                "company_size: str = Query(...)",
                "business_activities: Optional[List[str]] = Query(None)"
            ]
        },
        
        "get_industry_template": {
            "path": "/templates/industry/{industry}",
            "method": "GET",
            "description": "Get industry-specific compliance template",
            "parameters": [
                "industry: str",
                "authority: Optional[str] = Query(None)"
            ]
        },
        
        "get_big4_authorities": {
            "path": "/authorities/big4",
            "method": "GET", 
            "description": "Get Big 4 German authorities information",
            "parameters": []
        }
    }

# Integration code snippet for enhanced_compliance.py
INTEGRATION_CODE_SNIPPET = '''
# Add these imports to enhanced_compliance.py
from app.services.german_authority_engine.integration.authority_endpoints import (
    Big4AuthorityEndpoints, create_big4_authority_endpoints
)

# Add this after existing router initialization
big4_endpoints = create_big4_authority_endpoints()

# Add these new endpoints to your router

@router.post("/analyze-with-authority-detection")
async def analyze_with_authority_detection(
    files: List[UploadFile] = File(...),
    industry: Optional[str] = Form(None),
    company_location: Optional[str] = Form(None), 
    company_size: Optional[str] = Form(None),
    workspace_id: str = Form(default=DEMO_WORKSPACE_ID),
    db: AsyncSession = Depends(get_db_session)
):
    """Smart authority detection and comprehensive analysis"""
    return await big4_endpoints.analyze_with_smart_detection(
        files=files,
        industry=industry,
        company_location=company_location,
        company_size=company_size,
        workspace_id=workspace_id,
        db=db
    )

@router.post("/analyze-authority/{authority_id}")
async def analyze_authority_specific(
    authority_id: str,
    files: List[UploadFile] = File(...),
    industry: Optional[str] = Form(None),
    company_size: Optional[str] = Form(None),
    workspace_id: str = Form(default=DEMO_WORKSPACE_ID),
    db: AsyncSession = Depends(get_db_session)
):
    """Authority-specific compliance analysis"""
    return await big4_endpoints.analyze_for_specific_authority(
        authority_id=authority_id,
        files=files,
        industry=industry,
        company_size=company_size,
        workspace_id=workspace_id,
        db=db
    )

@router.post("/compare-authorities")
async def compare_authority_compliance(
    files: List[UploadFile] = File(...),
    authorities: List[str] = Form(...),
    industry: Optional[str] = Form(None),
    company_size: Optional[str] = Form(None),
    workspace_id: str = Form(default=DEMO_WORKSPACE_ID),
    db: AsyncSession = Depends(get_db_session)
):
    """Multi-authority compliance comparison"""
    return await big4_endpoints.compare_authorities(
        files=files,
        authorities=authorities,
        industry=industry,
        company_size=company_size,
        workspace_id=workspace_id,
        db=db
    )

@router.get("/authorities/detect-from-business")
async def detect_authorities_from_business(
    company_location: str = Query(...),
    industry: str = Query(...),
    company_size: str = Query(...),
    business_activities: Optional[List[str]] = Query(None)
):
    """Detect relevant authorities from business profile"""
    return await big4_endpoints.detect_relevant_authorities(
        company_location=company_location,
        industry=industry,
        company_size=company_size,
        business_activities=business_activities
    )

@router.get("/templates/industry/{industry}")
async def get_industry_compliance_template(
    industry: str,
    authority: Optional[str] = Query(None)
):
    """Get industry-specific compliance template"""
    return await big4_endpoints.get_industry_template(
        industry=industry,
        authority=authority
    )

@router.get("/authorities/big4")
async def get_big4_authorities():
    """Get Big 4 German authorities information"""
    return await big4_endpoints.get_all_big4_authorities_info()
'''