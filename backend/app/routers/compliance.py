# app/routers/compliance.py
from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from typing import List

from app.models.compliance import AnalysisResponse, ComplianceFramework
from app.services.file_processor import FileProcessor
from app.services.compliance_analyzer import ComplianceAnalyzer

router = APIRouter(prefix="/api/compliance", tags=["compliance"])

@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_compliance(
    files: List[UploadFile] = File(...),
    framework: str = Form(default="gdpr")
):
    """
    Analyze documents for compliance with specified framework
    Day 1: Basic analysis with German awareness
    Day 2: Enhanced with Docling and team features
    """
    
    # Validate framework
    try:
        compliance_framework = ComplianceFramework(framework.lower())
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported framework: {framework}. Supported: gdpr, soc2, hipaa, iso27001"
        )
    
    # Process files
    try:
        file_data = await FileProcessor.validate_and_extract(files)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"File processing failed: {str(e)}"
        )
    
    # Analyze for compliance
    try:
        analyzer = ComplianceAnalyzer()
        result = await analyzer.analyze_documents(file_data, compliance_framework)
        
        # Secure cleanup (GDPR compliance)
        for _, content, _ in file_data:
            await FileProcessor.secure_cleanup(content)
        
        return result
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Compliance analysis failed: {str(e)}"
        )

@router.get("/frameworks")
async def get_supported_frameworks():
    """Get list of supported compliance frameworks"""
    return {
        "supported_frameworks": [
            {
                "id": "gdpr",
                "name": "DSGVO / GDPR",
                "description": "EU-Datenschutzgrundverordnung / EU Data Protection Regulation",
                "region": "European Union",
                "german_support": True
            },
            {
                "id": "soc2",
                "name": "SOC 2",
                "description": "Security, availability, and confidentiality controls",
                "region": "Global (US-originated)",
                "german_support": False
            },
            {
                "id": "hipaa",
                "name": "HIPAA",
                "description": "Healthcare information privacy and security",
                "region": "United States",
                "german_support": False
            },
            {
                "id": "iso27001",
                "name": "ISO 27001",
                "description": "Information security management systems",
                "region": "Global",
                "german_support": False
            }
        ]
    }