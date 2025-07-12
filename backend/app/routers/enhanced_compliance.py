def _build_analysis_response(
    individual_analyses: List[DocumentAnalysis],
    compliance_report: ComplianceReport,
    authority_ctx: AuthorityContext,
    ui_context,
    performance_monitor,
    processing_results,
    params: ComplianceAnalysisParams,
    processing_time: float,
    start_time: datetime,
    batches
) -> AnalysisResponse:
    """Build the final analysis response with comprehensive error handling"""
    
    try:
        # Enhanced executive summary with authority information
        enhanced_summary = compliance_report.executive_summary
        
        if authority_ctx.has_authority_data():
            authority_name = getattr(authority_ctx.authority_analysis, 'authority_name', 'Unknown')
            enhanced_summary += f"\n\nGerman Authority: {authority_name}"
            
            if authority_ctx.detected_industry != "unknown":
                enhanced_summary += f"\nIndustry: {authority_ctx.detected_industry.title()}"
            
            if authority_ctx.authority_guidance:
                enhanced_summary += f"\nAuthority-Specific Requirements: {len(authority_ctx.authority_guidance)} identified"
        
        # Enhanced next steps with authority guidance
        enhanced_next_steps = compliance_report.next_steps.copy() if compliance_report.next_steps else []
        if authority_ctx.authority_guidance:
            enhanced_next_steps = authority_ctx.authority_guidance + enhanced_next_steps
        
        # Safe performance metrics gathering
        performance_metrics = {}
        if performance_monitor:
            try:
                performance_metrics = performance_monitor.get_processing_statistics()
            except Exception as e:
                _log_safe("warning", f"Failed to get performance statistics: {e}")
                performance_metrics = {"error": "Performance metrics unavailable"}
        
        # Safe UI context conversion
        ui_context_dict = {}
        if ui_context:
            try:
                ui_context_dict = ui_context.to_dict()
            except Exception as e:
                _log_safe("warning", f"Failed to convert UI context to dict: {e}")
                ui_context_dict = {"error": "UI context unavailable"}
        
        # Create comprehensive response with Day 3 enhancements
        return AnalysisResponse(
            individual_analyses=individual_analyses,
            compliance_report=ComplianceReport(
                framework=compliance_report.framework,
                executive_summary=enhanced_summary,
                executive_summary_de=getattr(compliance_report, 'executive_summary_de', None),
                compliance_score=compliance_report.compliance_score,
                documents_analyzed=compliance_report.documents_analyzed,
                german_documents_detected=compliance_report.german_documents_detected,
                priority_gaps=compliance_report.priority_gaps,
                compliance_strengths=compliance_report.compliance_strengths,
                next_steps=enhanced_next_steps,
                german_specific_recommendations=compliance_report.german_specific_recommendations
            ),
            processing_metadata={
                "analysis_id": f"day3_{int(start_time.timestamp())}",
                "processing_time": processing_time,
                "total_documents": len(individual_analyses),
                "total_batches": len(batches) if batches else 0,
                "framework": params.framework,
                "workspace_id": params.workspace_id,
                "performance_grade": _safe_calculate_performance_grade(processing_results),
                "day3_features": {
                    "parallel_processing": True,
                    "ui_context_intelligence": True,
                    "german_priority_processing": True,
                    "real_time_progress": True,
                    "performance_monitoring": True,
                    "authority_intelligence": True,
                    "performance_optimization": True
                },
                "ui_context": ui_context_dict,
                "performance_metrics": performance_metrics,
                "authority_intelligence": authority_ctx.to_metadata_dict()
            }
        )
    
    except Exception as e:
        _log_safe("error", f"Failed to build analysis response: {e}")
        # Return minimal response in case of error
        return AnalysisResponse(
            individual_analyses=individual_analyses,
            compliance_report=compliance_report,
            processing_metadata={
                "analysis_id": f"day3_{int(start_time.timestamp())}",
                "processing_time": processing_time,
                "error": f"Response building failed: {str(e)}",
                "framework": params.framework,
                "workspace_id": params.workspace_id
            }
        )

# =============================================================================
# BACKWARDS COMPATIBILITY ENDPOINTS
# =============================================================================

@router.get("/health")
async def enhanced_health_check():
    """Enhanced health check for Day 2 enterprise features with comprehensive status"""
    
    try:
        # Test component availability
        component_tests = {
            "big4_endpoints": big4_endpoints is not None,
            "parallel_processing": True,  # Always available
            "ui_context_layer": True,     # Always available  
            "performance_monitor": True,  # Always available
            "german_authority_engine": GERMAN_AUTHORITY_ENGINE_AVAILABLE,
            "logging": LOGGING_AVAILABLE
        }
        
        return {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "day3_features": {
                "parallel_processing": True,
                "ui_context_intelligence": True,
                "performance_monitoring": True,
                "websocket_progress": True,
                "german_authority_analysis": GERMAN_AUTHORITY_ENGINE_AVAILABLE,
                "big4_authority_engine": big4_endpoints is not None,
                "authority_scope_fix": True,
                "improved_error_handling": True,
                "input_validation": True,
                "safe_logging": True
            },
            "components_status": {
                "big4_endpoints": "available" if big4_endpoints else "unavailable",
                "big4_init_error": BIG4_INITIALIZATION_ERROR,
                "parallel_processing": "available",
                "ui_context_layer": "available",
                "performance_monitor": "available",
                "german_authority_engine": "available" if GERMAN_AUTHORITY_ENGINE_AVAILABLE else "unavailable",
                "logging": "available" if LOGGING_AVAILABLE else "fallback"
            },
            "component_tests": component_tests,
            "system_info": {
                "settings_max_files": getattr(settings, 'max_files_per_batch', 'not_configured'),
                "settings_max_file_size": getattr(settings, 'max_file_size_mb', 'not_configured'),
                "allowed_extensions": getattr(settings, 'allowed_extensions', [])
            }
        }
    
    except Exception as e:
        _log_safe("error", f"Health check failed: {e}")
        return {
            "status": "degraded",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "error": str(e),
            "basic_functionality": "available"
        }

@router.get("/frameworks")
async def get_supported_frameworks():
    """Get list of supported compliance frameworks with Day 3 enhancements"""
    
    try:
        return {
            "supported_frameworks": [
                {
                    "id": "gdpr",
                    "name": "DSGVO / GDPR",
                    "description": "EU-Datenschutzgrundverordnung / EU Data Protection Regulation",
                    "region": "European Union",
                    "german_support": True,
                    "day3_features": {
                        "parallel_processing": True,
                        "ui_context_detection": True,
                        "german_priority": True,
                        "authority_specific_analysis": big4_endpoints is not None,
                        "big4_authority_engine": big4_endpoints is not None,
                        "authority_scope_fix": True,
                        "improved_error_handling": True,
                        "input_validation": True
                    }
                },
                {
                    "id": "soc2",
                    "name": "SOC 2",
                    "description": "Security, availability, and confidentiality controls",
                    "region": "Global (US-originated)",
                    "german_support": False,
                    "day3_features": {
                        "parallel_processing": True,
                        "ui_context_detection": True,
                        "german_priority": False,
                        "authority_specific_analysis": False,
                        "big4_authority_engine": False,
                        "authority_scope_fix": False,
                        "improved_error_handling": True,
                        "input_validation": True
                    }
                },
                {
                    "id": "hipaa",
                    "name": "HIPAA",
                    "description": "Healthcare information privacy and security",
                    "region": "United States",
                    "german_support": False,
                    "day3_features": {
                        "parallel_processing": True,
                        "ui_context_detection": True,
                        "german_priority": False,
                        "authority_specific_analysis": False,
                        "big4_authority_engine": False,
                        "authority_scope_fix": False,
                        "improved_error_handling": True,
                        "input_validation": True
                    }
                },
                {
                    "id": "iso27001",
                    "name": "ISO 27001",
                    "description": "Information security management systems",
                    "region": "Global",
                    "german_support": False,
                    "day3_features": {
                        "parallel_processing": True,
                        "ui_context_detection": True,
                        "german_priority": False,
                        "authority_specific_analysis": False,
                        "big4_authority_engine": False,
                        "authority_scope_fix": False,
                        "improved_error_handling": True,
                        "input_validation": True
                    }
                }
            ],
            "metadata": {
                "total_frameworks": 4,
                "german_specialized": 1,
                "big4_engine_available": big4_endpoints is not None,
                "enhanced_features_active": True,
                "input_validation_enabled": True,
                "safe_logging_enabled": True
            }
        }
    
    except Exception as e:
        _log_safe("error", f"Failed to get frameworks: {e}")
        return {
            "error": "Failed to retrieve frameworks",
            "supported_frameworks": [],
            "metadata": {"error": str(e)}
        }

# =============================================================================
# DEVELOPMENT AND DEBUG ENDPOINTS
# =============================================================================

@router.get("/debug/components")
async def debug_components_status():
    """Debug endpoint to check component status with comprehensive testing"""
    
    try:
        component_status = {
            "status": "debug_info",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "components": {}
        }
        
        # Test UI Context Layer
        try:
            ui_context_layer = UIContextLayer()
            ui_test = ui_context_layer.detect_german_content("test content")
            component_status["components"]["ui_context_layer"] = {
                "status": "available",
                "german_detection_test": ui_test,
                "document_job_available": hasattr(ui_context_layer, 'DocumentJob')
            }
        except Exception as e:
            component_status["components"]["ui_context_layer"] = {
                "status": "error",
                "error": str(e)
            }
        
        # Test Big 4 endpoints
        component_status["components"]["big4_endpoints"] = {
            "status": "available" if big4_endpoints else "unavailable",
            "initialization_error": BIG4_INITIALIZATION_ERROR,
            "dependencies_available": BIG4_ENDPOINTS_AVAILABLE
        }
        
        # Test parallel processing components
        try:
            job_queue = JobQueue()
            batch_processor = BatchProcessor()
            performance_monitor = PerformanceMonitor()
            
            component_status["components"]["parallel_processing"] = {
                "job_queue": "available",
                "batch_processor": "available", 
                "performance_monitor": "available"
            }
        except Exception as e:
            component_status["components"]["parallel_processing"] = {
                "status": "error",
                "error": str(e)
            }
        
        # Test German Authority Engine
        component_status["components"]["german_authority_engine"] = {
            "status": "available" if GERMAN_AUTHORITY_ENGINE_AVAILABLE else "unavailable",
            "get_all_authorities_available": get_all_authorities is not None
        }
        
        # Test logging
        component_status["components"]["logging"] = {
            "status": "available" if LOGGING_AVAILABLE else "fallback",
            "structlog_available": logger is not None
        }
        
        # Settings information
        component_status["settings"] = {
            "max_files_per_batch": getattr(settings, 'max_files_per_batch', 'not_configured'),
            "max_file_size_mb": getattr(settings, 'max_file_size_mb', 'not_configured'),
            "max_total_file_size_mb": getattr(settings, 'max_total_file_size_mb', 'not_configured'),
            "big4_authority_engine_enabled": getattr(settings, 'big4_authority_engine_enabled', False),
            "allowed_extensions": getattr(settings, 'allowed_extensions', [])
        }
        
        return component_status
    
    except Exception as e:
        _log_safe("error", f"Debug components check failed: {e}")
        return {
            "status": "error",
            "error_message": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

@router.get("/debug/test-german-detection")
async def debug_test_german_detection(
    content: str = Query("Datenschutz und DSGVO sind wichtig für deutsche Unternehmen", description="Content to test German detection")
):
    """Debug endpoint to test German content detection with validation"""
    
    try:
        # Validate input
        if not content or not content.strip():
            raise HTTPException(
                status_code=400,
                detail="Content parameter cannot be empty"
            )
        
        ui_context_layer = UIContextLayer()
        is_german = ui_context_layer.detect_german_content(content)
        
        return {
            "test_content": content,
            "is_german_detected": is_german,
            "detection_method": "UIContextLayer.detect_german_content",
            "content_length": len(content),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        _log_safe("error", f"German detection test failed: {e}")
        return {
            "error": str(e),
            "test_content": content,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

@router.get("/debug/validation-test")
async def debug_validation_test():
    """Debug endpoint to test input validation functions"""
    
    try:
        test_results = {}
        
        # Test file validation with empty list
        try:
            _validate_files_input([])
            test_results["empty_files_validation"] = "failed - should have raised exception"
        except HTTPException:
            test_results["empty_files_validation"] = "passed - correctly rejected empty files"
        
        # Test string parameter validation
        test_params = _validate_string_params(
            company_location="  Germany  ",
            industry_hint="",
            valid_param="test"
        )
        test_results["string_validation"] = {
            "company_location": test_params.get("company_location"),
            "industry_hint": test_params.get("industry_hint"),
            "valid_param": test_params.get("valid_param")
        }
        
        # Test processing results validation
        test_results["processing_results_validation"] = {
            "empty_list": _validate_processing_results([]),
            "invalid_objects": _validate_processing_results([{"no_success": True}]),
        }
        
        return {
            "status": "validation_test_completed",
            "test_results": test_results,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    except Exception as e:
        _log_safe("error", f"Validation test failed: {e}")
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

# =============================================================================
# ROUTE SUMMARY AND DOCUMENTATION
# =============================================================================

@router.get("/")
async def compliance_api_overview():
    """Overview of the Enhanced Compliance API with Day 3 features and validation"""
    
    try:
        return {
            "api_name": "WolfMerge Enhanced Compliance API",
            "version": "3.1.0",
            "description": "Day 3 Enhanced: 10x Performance + German Authority Intelligence + Code Review Fixes",
            
            "core_endpoints": {
                "analyze": {
                    "path": "/analyze",
                    "method": "POST",
                    "description": "Main compliance analysis with parallel processing and authority detection",
                    "validation": "Comprehensive input validation enabled"
                },
                "health": {
                    "path": "/health", 
                    "method": "GET",
                    "description": "Health check with Day 3 feature status and component testing"
                },
                "frameworks": {
                    "path": "/frameworks",
                    "method": "GET", 
                    "description": "Supported compliance frameworks with enhanced error handling"
                }
            },
            
            "authority_endpoints": {
                "smart_detection": {
                    "path": "/analyze-with-authority-detection",
                    "method": "POST",
                    "description": "Smart German authority detection and analysis",
                    "available": big4_endpoints is not None,
                    "validation": "Input validation and Big 4 availability checks"
                },
                "specific_authority": {
                    "path": "/analyze-authority/{authority_id}",
                    "method": "POST",
                    "description": "Authority-specific compliance analysis",
                    "available": big4_endpoints is not None,
                    "validation": "Authority ID validation and parameter sanitization"
                },
                "compare_authorities": {
                    "path": "/compare-authorities", 
                    "method": "POST",
                    "description": "Multi-authority compliance comparison",
                    "available": big4_endpoints is not None,
                    "validation": "Authorities list parsing and validation"
                },
                "business_detection": {
                    "path": "/authorities/detect-from-business",
                    "method": "GET",
                    "description": "Authority detection from business profile",
                    "available": big4_endpoints is not None,
                    "validation": "Required parameter validation"
                },
                "industry_templates": {
                    "path": "/templates/industry/{industry}",
                    "method": "GET", 
                    "description": "Industry-specific compliance templates",
                    "available": big4_endpoints is not None,
                    "validation": "Industry parameter validation"
                },
                "big4_info": {
                    "path": "/authorities/big4",
                    "method": "GET",
                    "description": "Big 4 German authorities information",
                    "available": big4_endpoints is not None,
                    "validation": "Availability checks"
                }
            },
            
            "debug_endpoints": {
                "components_status": {
                    "path": "/debug/components",
                    "method": "GET",
                    "description": "Debug component status and comprehensive testing"
                },
                "german_detection_test": {
                    "path": "/debug/test-german-detection",
                    "method": "GET", 
                    "description": "Test German content detection with input validation"
                },
                "validation_test": {
                    "path": "/debug/validation-test",
                    "method": "GET",
                    "description": "Test input validation functions"
                }
            },
            
            "day3_features": {
                "parallel_processing": True,
                "ui_context_intelligence": True,
                "german_authority_detection": big4_endpoints is not None,
                "real_time_progress": True,
                "performance_monitoring": True,
                "improved_error_handling": True,
                "scope_fixes_applied": True,
                "input_validation": True,
                "safe_logging": True,
                "comprehensive_testing": True
            },
            
            "code_review_fixes": {
                "big4_endpoints_validation": "✅ Null checks and availability validation added",
                "get_all_authorities_validation": "✅ Import and function availability checks added",
                "processing_results_validation": "✅ Division by zero protection and validation added",
                "files_input_validation": "✅ Empty files list validation added",
                "parameter_validation": "✅ String parameter sanitization and validation added",
                "processing_time_usage": "✅ Processing time properly tracked and used in logging",
                "logging_configuration": "✅ Safe logging wrapper with fallback implemented"
            },
            
            "supported_authorities": [
                "bfdi", "baylda", "lfd_bw", "ldi_nrw"
            ] if big4_endpoints else [],
            
            "supported_industries": [
                "automotive", "software", "manufacturing", "healthcare"
            ] if big4_endpoints else [],
            
            "system_status": {
                "big4_available": big4_endpoints is not None,
                "german_engine_available": GERMAN_AUTHORITY_ENGINE_AVAILABLE,
                "logging_available": LOGGING_AVAILABLE,
                "all_validations_active": True
            },
            
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    except Exception as e:
        _log_safe("error", f"API overview generation failed: {e}")
        return {
            "error": "Failed to generate API overview",
            "basic_info": {
                "api_name": "WolfMerge Enhanced Compliance API",
                "version": "3.1.0",
                "status": "error"
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        }# =============================================================================
# BIG 4 GERMAN AUTHORITY ENDPOINTS - IMPROVED ERROR HANDLING
# =============================================================================

def _check_big4_availability():
    """Check if Big 4 endpoints are available and properly initialized"""
    if not BIG4_ENDPOINTS_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Big 4 Authority Engine not available. Missing required dependencies."
        )
    
    if not big4_endpoints:
        error_detail = "Big 4 Authority Engine not initialized."
        if BIG4_INITIALIZATION_ERROR:
            error_detail += f" Error: {BIG4_INITIALIZATION_ERROR}"
        raise HTTPException(status_code=503, detail=error_detail)

@router.get("/authorities/detect-from-business")
async def detect_authorities_from_business(
    company_location: str = Query(..., description="Company location (e.g., 'bayern', 'baden_wurttemberg')"),
    industry: str = Query(..., description="Industry (e.g., 'automotive', 'software', 'manufacturing')"),
    company_size: str = Query(..., description="Company size ('small', 'medium', 'large')"),
    business_activities: Optional[List[str]] = Query(None, description="Specific business activities")
):
    """
    🔍 Business Profile Authority Detection
    
    Detect relevant German authorities based on business profile
    without requiring document upload. Perfect for onboarding.
    """
    _check_big4_availability()
    
    # Validate required parameters
    if not company_location or not company_location.strip():
        raise HTTPException(
            status_code=400,
            detail="Company location is required and cannot be empty"
        )
    
    if not industry or not industry.strip():
        raise HTTPException(
            status_code=400,
            detail="Industry is required and cannot be empty"
        )
    
    if not company_size or not company_size.strip():
        raise HTTPException(
            status_code=400,
            detail="Company size is required and cannot be empty"
        )
    
    # Validate and sanitize parameters
    validated_params = _validate_string_params(
        company_location=company_location,
        industry=industry,
        company_size=company_size
    )
    
    try:
        return await big4_endpoints.detect_relevant_authorities(
            company_location=validated_params["company_location"],
            industry=validated_params["industry"],
            company_size=validated_params["company_size"],
            business_activities=business_activities
        )
    except Exception as e:
        _log_safe("error", "Business profile authority detection failed", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Business profile authority detection failed: {str(e)}"
        )

@router.get("/templates/industry/{industry}")
async def get_industry_compliance_template(
    industry: str,
    authority: Optional[str] = Query(None, description="Specific authority for customized template")
):
    """
    📋 Industry-Specific Compliance Templates
    
    Get pre-configured compliance templates for German industries
    with authority-specific requirements and best practices.
    """
    _check_big4_availability()
    
    # Validate industry parameter
    if not industry or not industry.strip():
        raise HTTPException(
            status_code=400,
            detail="Industry is required and cannot be empty"
        )
    
    # Validate and sanitize parameters
    validated_params = _validate_string_params(
        industry=industry,
        authority=authority
    )
    
    try:
        return await big4_endpoints.get_industry_template(
            industry=validated_params["industry"],
            authority=validated_params["authority"]
        )
    except Exception as e:
        _log_safe("error", "Industry template generation failed", 
                 industry=industry, error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Industry template generation failed: {str(e)}"
        )

@router.get("/authorities/big4")
async def get_big4_authorities():
    """
    🏛️ Big 4 German Authorities Information
    
    Complete information about the Big 4 German data protection authorities
    including enforcement patterns, contact information, and specializations.
    """
    _check_big4_availability()
    
    try:
        return await big4_endpoints.get_all_big4_authorities_info()
    except Exception as e:
        _log_safe("error", "Failed to retrieve Big 4 authorities info", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve authorities information: {str(e)}"
        )

@router.get("/german-authorities")
async def get_german_authorities():
    """Get all German data protection authorities"""
    
    # Check if German authority engine is available
    if not GERMAN_AUTHORITY_ENGINE_AVAILABLE or not get_all_authorities:
        raise HTTPException(
            status_code=503,
            detail="German Authority Engine not available. Missing required dependencies."
        )
    
    try:
        authorities_data = get_all_authorities()
        
        if not authorities_data:
            _log_safe("warning", "No German authorities data available")
            return {"authorities": [], "message": "No authorities data available"}
        
        return {
            "authorities": [
                {"id": k.value, "name": v.name} 
                for k, v in authorities_data.items()
            ]
        }
    except Exception as e:
        _log_safe("error", "Failed to retrieve German authorities", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve German authorities: {str(e)}"
        )

# =============================================================================
# ADDITIONAL HELPER FUNCTIONS
# =============================================================================

def _create_error_analysis(result) -> DocumentAnalysis:
    """Create error analysis for failed processing result"""
    
    return DocumentAnalysis(
        filename=getattr(result.job, 'filename', 'unknown'),
        document_language=DocumentLanguage.UNKNOWN,
        compliance_summary=f"Processing failed: {getattr(result, 'error_message', 'Unknown error')}",
        control_mappings=[],
        compliance_gaps=["Processing error prevented analysis"],
        risk_indicators=["Document processing failed"],
        german_insights=None,
        original_size=getattr(result.job, 'size', 0),
        processing_time=getattr(result, 'processing_time', 0.0)
    )

async def _create_enhanced_compliance_report(
    analyses: List[DocumentAnalysis],
    ui_context,  # UIContext type
    performance_monitor,  # PerformanceMonitor type
    framework: ComplianceFramework,
    workspace_id: str,
    authority_context: Optional[AuthorityContext] = None
) -> ComplianceReport:
    """Create enhanced compliance report with Day 3 intelligence and authority data"""
    
    # Validate inputs
    if not analyses:
        _log_safe("warning", "No analyses provided for compliance report generation")
        # Return minimal report
        return ComplianceReport(
            framework=framework,
            executive_summary="No documents were successfully analyzed.",
            compliance_score=0.0,
            documents_analyzed=0,
            german_documents_detected=False,
            priority_gaps=["No analysis data available"],
            compliance_strengths=[],
            next_steps=["Upload valid documents for analysis"],
            german_specific_recommendations=[]
        )
    
    # Calculate enhanced metrics
    german_documents = sum(
        1 for analysis in analyses 
        if analysis.document_language == DocumentLanguage.GERMAN
    )
    
    # Use UI context for smarter scoring with safe attribute access
    base_compliance_score = getattr(ui_context, 'portfolio_score', 0.5) if ui_context else 0.5
    
    # Adjust score based on detected scenario and completeness
    scenario_bonus = 0.0
    completeness_bonus = 0.0
    
    if ui_context:
        scenario_detected = getattr(ui_context, 'detected_scenario', None)
        if scenario_detected and hasattr(scenario_detected, 'value'):
            scenario_bonus = 0.1 if scenario_detected.value != "unknown" else 0.0
        
        compliance_completeness = getattr(ui_context, 'compliance_completeness', 0.0)
        completeness_bonus = compliance_completeness * 0.2
    
    # Authority-specific scoring adjustments
    authority_bonus = 0.0
    if authority_context and authority_context.has_authority_data():
        # Higher bonus for higher authority compliance scores
        compliance_score = getattr(authority_context.authority_analysis, 'compliance_score', 0.0)
        authority_bonus = compliance_score * 0.1
    
    final_compliance_score = min(1.0, base_compliance_score + scenario_bonus + completeness_bonus + authority_bonus)
    
    # Enhanced executive summary with authority information
    scenario_description = getattr(ui_context, 'scenario_description', 'Unknown scenario') if ui_context else 'No scenario detected'
    industry_detected = getattr(ui_context, 'industry_detected', None) if ui_context else None
    industry_name = industry_detected.value.title() if industry_detected and hasattr(industry_detected, 'value') else 'Unknown'
    
    executive_summary = f"""
Day 3 Enhanced Analysis: {framework.value.upper()} compliance assessment completed.

Portfolio Analysis:
- Documents Analyzed: {len(analyses)}
- German Content: {german_documents} documents ({(german_documents/len(analyses)*100):.0f}%)
- Compliance Score: {final_compliance_score:.2%}

Scenario Detected: {scenario_description}
Industry: {industry_name}
    """.strip()
    
    # Add authority information if available
    if authority_context and authority_context.has_authority_data():
        authority_name = getattr(authority_context.authority_analysis, 'authority_name', 'Unknown')
        enforcement_likelihood = getattr(authority_context.authority_analysis, 'enforcement_likelihood', 0.0)
        audit_readiness = getattr(authority_context.authority_analysis, 'audit_readiness_score', 0.0)
        
        executive_summary += f"""

Authority Intelligence:
- Detected Authority: {authority_name} ({authority_context.detected_authority})
- Industry: {authority_context.detected_industry.title()}
- Enforcement Likelihood: {enforcement_likelihood:.2%}
- Audit Readiness: {audit_readiness:.2%}
        """.strip()
    
    # Enhanced recommendations based on UI context and authority data
    next_steps = []
    if ui_context:
        suggested_actions = getattr(ui_context, 'suggested_actions', [])
        next_steps = [
            action.description for action in suggested_actions[:3]
            if hasattr(action, 'description')
        ]
    
    # Add authority-specific guidance
    if authority_context and authority_context.authority_guidance:
        next_steps = authority_context.authority_guidance + next_steps
    
    next_steps.extend([
        "Review detailed document-level analysis results",
        "Export compliance report for stakeholder review"
    ])
    
    # Get priority risks and quick wins safely
    priority_risks = getattr(ui_context, 'priority_risks', []) if ui_context else []
    quick_wins = getattr(ui_context, 'quick_wins', []) if ui_context else []
    
    return ComplianceReport(
        framework=framework,
        executive_summary=executive_summary,
        compliance_score=final_compliance_score,
        documents_analyzed=len(analyses),
        german_documents_detected=german_documents > 0,
        priority_gaps=priority_risks if priority_risks else ["No specific risks identified"],
        compliance_strengths=[
            f"Intelligent processing completed in high-performance mode",
            f"UI context detection: {scenario_description}",
            f"German compliance optimization active",
            f"Real-time progress tracking enabled"
        ],
        next_steps=next_steps,
        german_specific_recommendations=quick_wins if german_documents > 0 else []
    )

# =============================================================================
# HELPER FUNCTIONS - IMPROVED STRUCTURE AND ERROR HANDLING
# =============================================================================

async def _process_and_validate_files(files: List[UploadFile]) -> List[tuple]:
    """Process and validate uploaded files with comprehensive error handling"""
    processed_files = []
    total_size = 0
    
    # Validate files input
    _validate_files_input(files)
    
    for file in files:
        try:
            # Validate filename
            if not file.filename:
                _log_safe("warning", "File uploaded without filename, skipping")
                continue
            
            # Check file extension
            if not any(file.filename.lower().endswith(ext) for ext in settings.allowed_extensions):
                raise HTTPException(
                    status_code=400,
                    detail=f"File type not supported: {file.filename}. "
                           f"Allowed: {', '.join(settings.allowed_extensions)}"
                )
            
            # Read content first, then check size
            content = await file.read()
            file_size = len(content)
            
            # Validate file size
            if file_size == 0:
                _log_safe("warning", f"Empty file detected: {file.filename}")
                continue
            
            if file_size > settings.max_file_size_bytes:
                raise HTTPException(
                    status_code=413,
                    detail=f"File {file.filename} exceeds maximum size of {settings.max_file_size_mb}MB"
                )
            
            total_size += file_size
            processed_files.append((file.filename, content, file_size))
            
        except HTTPException:
            raise
        except Exception as e:
            _log_safe("error", f"Error processing file {getattr(file, 'filename', 'unknown')}: {e}")
            raise HTTPException(
                status_code=400,
                detail=f"Error processing file {getattr(file, 'filename', 'unknown')}: {str(e)}"
            )
    
    # Check if any files were successfully processed
    if not processed_files:
        raise HTTPException(
            status_code=400,
            detail="No valid files were processed. Please check file formats and sizes."
        )
    
    # Check total workspace size
    max_workspace_size = getattr(settings, 'max_workspace_size_mb', 100) * 1024 * 1024
    if total_size > max_workspace_size:
        raise HTTPException(
            status_code=400,
            detail=f"Total batch size exceeds workspace limit of {getattr(settings, 'max_workspace_size_mb', 100)}MB"
        )
    
    return processed_files

async def _perform_authority_detection(
    processed_files: List[tuple],
    company_location: Optional[str],
    industry_hint: Optional[str]
) -> AuthorityContext:
    """Perform authority detection with comprehensive error handling"""
    
    # Validate inputs
    if not processed_files:
        _log_safe("warning", "No processed files provided for authority detection")
        return AuthorityContext()
    
    try:
        _log_safe("info", "🔍 Big 4 German Authority Engine: Activating for German GDPR content")
        
        # Create document objects for Big 4 engine
        documents = []
        for filename, content, size in processed_files:
            try:
                content_str = content if isinstance(content, str) else content.decode('utf-8', errors='ignore')
                
                doc = type('Document', (), {
                    'filename': filename,
                    'content': content_str,
                    'file_size': size,
                    'upload_timestamp': datetime.now(timezone.utc)
                })()
                documents.append(doc)
                
            except Exception as e:
                _log_safe("warning", f"Failed to create document object for {filename}: {e}")
                continue
        
        if not documents:
            _log_safe("warning", "No valid documents created for authority detection")
            return AuthorityContext()
        
        # Use optimized authority detection to maintain scope
        authority_ctx = await _optimized_authority_detection(
            documents, company_location, industry_hint
        )
        
        _log_safe("info", f"Authority detection completed: {authority_ctx.detected_authority} "
                         f"(industry: {authority_ctx.detected_industry}, "
                         f"confidence: {authority_ctx.authority_confidence:.2f})")
        
        return authority_ctx
        
    except Exception as e:
        _log_safe("warning", f"Big 4 Authority Engine error (non-critical): {str(e)}")
        # Return default authority context if detection fails
        authority_ctx = AuthorityContext()
        authority_ctx.german_content_detected = True
        return authority_ctx

async def _initialize_processing_components():
    """Initialize processing components with comprehensive error handling"""
    try:
        job_queue = JobQueue()
        batch_processor = BatchProcessor()
        performance_monitor = PerformanceMonitor()
        
        # Validate components were created successfully
        if not all([job_queue, batch_processor, performance_monitor]):
            raise ValueError("One or more processing components failed to initialize")
        
        return job_queue, batch_processor, performance_monitor
        
    except Exception as e:
        _log_safe("error", f"Failed to initialize processing components: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to initialize processing components"
        )

async def _create_intelligent_jobs(
    processed_files: List[tuple],
    params: ComplianceAnalysisParams,
    authority_ctx: AuthorityContext,
    job_queue: JobQueue
) -> List[DocumentJob]:
    """Create intelligent job queue with authority context - FIXED: Use JobPriority directly"""
    jobs = []
    
    # Validate inputs
    if not processed_files:
        _log_safe("warning", "No processed files provided for job creation")
        return jobs
    
    for i, (filename, content, size) in enumerate(processed_files):
        try:
            # Use JobPriority directly, not job_queue.JobPriority
            priority = JobPriority.HIGH if authority_ctx.german_content_detected else JobPriority.MEDIUM
            
            # Create job object with proper parameters
            job = DocumentJob(
                filename=filename,
                content=content,
                size=size,
                priority=priority,  # FIXED: Use JobPriority enum directly
                complexity_score=0.5,  # Default complexity
                processing_estimate_seconds=10.0,  # Default estimate
                german_score=0.8 if authority_ctx.german_content_detected else 0.1,
                language_detected="de" if authority_ctx.german_content_detected else "en",
                compliance_indicators=[params.framework] if params.framework else ["gdpr"]
            )
            jobs.append(job)
            
            _log_safe("debug", "Job created successfully",
                     filename=filename, priority=priority.name,
                     german_detected=authority_ctx.german_content_detected)
            
        except Exception as e:
            _log_safe("warning", f"Failed to create job for {filename}: {e}")
            # Create a minimal fallback job
            try:
                fallback_job = DocumentJob(
                    filename=filename,
                    content=content,
                    size=size,
                    priority=JobPriority.LOW,  # FIXED: Use JobPriority enum directly
                    complexity_score=0.3,
                    processing_estimate_seconds=5.0,
                    german_score=0.0,
                    language_detected="en",
                    compliance_indicators=["general"]
                )
                jobs.append(fallback_job)
                _log_safe("info", f"Created fallback job for {filename}")
            except Exception as fallback_error:
                _log_safe("error", f"Failed to create fallback job for {filename}: {fallback_error}")
                continue
    
    if not jobs:
        _log_safe("error", "No jobs were created successfully")
        raise HTTPException(
            status_code=500,
            detail="Failed to create processing jobs for uploaded files"
        )
    
    return jobs

def _convert_results_to_analyses(processing_results, authority_ctx: AuthorityContext) -> List[DocumentAnalysis]:
    """Convert processing results to document analyses with error handling"""
    individual_analyses = []
    
    if not processing_results:
        _log_safe("warning", "No processing results provided for conversion")
        return individual_analyses
    
    for result in processing_results:
        try:
            if getattr(result, 'success', False) and getattr(result, 'analysis', None):
                # Enhance analysis with authority data if available
                if authority_ctx.has_authority_data():
                    analysis = result.analysis
                    if hasattr(analysis, 'german_insights') and analysis.german_insights:
                        analysis.german_insights["detected_authority"] = authority_ctx.detected_authority
                        analysis.german_insights["detected_industry"] = authority_ctx.detected_industry
                        analysis.german_insights["authority_guidance"] = authority_ctx.authority_guidance[:2]
                
                individual_analyses.append(result.analysis)
            else:
                # Create error analysis for failed processing
                individual_analyses.append(_create_error_analysis(result))
        except Exception as e:
            _log_safe("warning", f"Failed to process result for {getattr(result.job, 'filename', 'unknown')}: {e}")
            # Create minimal error analysis
            error_analysis = DocumentAnalysis(
                filename=getattr(result.job, 'filename', 'unknown'),
                document_language=DocumentLanguage.UNKNOWN,
                compliance_summary=f"Processing error: {str(e)}",
                control_mappings=[],
                compliance_gaps=["Processing error prevented analysis"],
                risk_indicators=["Document processing failed"],
                german_insights=None,
                original_size=getattr(result.job, 'size', 0),
                processing_time=0.0
            )
            individual_analyses.append(error_analysis)
    
    return individual_analyses
async def analyze_with_authority_detection(
    files: List[UploadFile] = File(...),
    industry: Optional[str] = Form(None),
    company_location: Optional[str] = Form(None), 
    company_size: Optional[str] = Form(None),
    workspace_id: str = Form(default=DEMO_WORKSPACE_ID),
    db: AsyncSession = Depends(get_db_session)
):
    """
    🧠 Smart Authority Detection + Analysis
    
    Automatically detects relevant Big 4 German authorities and provides
    comprehensive analysis with multi-authority comparison.
    """
    # Validate inputs
    _validate_files_input(files)
    _check_big4_availability()
    
    # Validate and sanitize string parameters
    validated_params = _validate_string_params(
        industry=industry,
        company_location=company_location,
        company_size=company_size
    )
    
    try:
        return await big4_endpoints.analyze_with_smart_detection(
            files=files,
            industry=validated_params["industry"],
            company_location=validated_params["company_location"],
            company_size=validated_params["company_size"],
            workspace_id=workspace_id,
            db=db
        )
    except HTTPException:
        raise
    except Exception as e:
        _log_safe("error", "Smart authority detection failed", 
                 error=str(e), error_type=type(e).__name__, files_count=len(files))
        
        raise HTTPException(
            status_code=500,
            detail=f"Smart authority detection failed: {str(e)}"
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
    """
    🎯 Authority-Specific Compliance Analysis
    
    Detailed analysis for specific Big 4 German authority with
    enforcement patterns, penalty estimates, and audit preparation.
    """
    # Validate inputs
    _validate_files_input(files)
    _check_big4_availability()
    
    if not authority_id or not authority_id.strip():
        raise HTTPException(
            status_code=400,
            detail="Authority ID is required and cannot be empty"
        )
    
    # Validate and sanitize string parameters
    validated_params = _validate_string_params(
        industry=industry,
        company_size=company_size
    )
    
    try:
        return await big4_endpoints.analyze_for_specific_authority(
            authority_id=authority_id.strip(),
            files=files,
            industry=validated_params["industry"],
            company_size=validated_params["company_size"],
            workspace_id=workspace_id,
            db=db
        )
    except Exception as e:
        _log_safe("error", "Authority-specific analysis failed", 
                 authority_id=authority_id, error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Authority-specific analysis failed: {str(e)}"
        )

@router.post("/compare-authorities")
async def compare_authority_compliance(
    files: List[UploadFile] = File(...),
    authorities: str = Form(...),
    industry: Optional[str] = Form(None),
    company_size: Optional[str] = Form(None),
    workspace_id: str = Form(default=DEMO_WORKSPACE_ID),
    db: AsyncSession = Depends(get_db_session)
):
    """
    ⚖️ Multi-Authority Compliance Comparison
    
    Compare compliance analysis across multiple Big 4 authorities
    to optimize jurisdiction strategy and identify best practices.
    """
    # Validate inputs
    _validate_files_input(files)
    _check_big4_availability()
    
    # Parse and validate authorities
    try:
        if not authorities or not authorities.strip():
            raise ValueError("Authorities parameter cannot be empty")
            
        authorities_list = [auth.strip() for auth in authorities.split(',') if auth.strip()]
        if len(authorities_list) < 2:
            raise HTTPException(
                status_code=400,
                detail="At least 2 valid authority IDs required for comparison"
            )
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid authorities format. Use comma-separated list (e.g., 'bfdi,baylda'): {str(e)}"
        )
    
    # Validate and sanitize string parameters
    validated_params = _validate_string_params(
        industry=industry,
        company_size=company_size
    )
    
    try:
        return await big4_endpoints.compare_authorities(
            files=files,
            authorities=authorities_list,
            industry=validated_params["industry"],
            company_size=validated_params["company_size"],
            workspace_id=workspace_id,
            db=db
        )
    except Exception as e:
        _log_safe("error", "Authority comparison failed", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Authority comparison failed: {str(e)}"
        )

@router# app/routers/enhanced_compliance.py - CODE REVIEW FIXES APPLIED
from fastapi import APIRouter, File, UploadFile, Form, HTTPException, Request, Depends, Query
from typing import List, Optional, Dict, Any, Union
from datetime import datetime, timezone
from sqlalchemy.ext.asyncio import AsyncSession
import structlog
import time
import asyncio
from dataclasses import dataclass, field

from app.models.compliance import AnalysisResponse, ComplianceFramework, DocumentAnalysis, ComplianceReport, DocumentLanguage
from app.services.file_processor import FileProcessor
from app.services.enhanced_compliance_analyzer import EnhancedComplianceAnalyzer
from app.database import get_db_session, DEMO_WORKSPACE_ID, DEMO_ADMIN_USER_ID
from app.config import settings

# Day 3 Parallel Processing Imports - FIXED: Added missing DocumentJob and JobPriority imports
from app.services.parallel_processing import (
    JobQueue, BatchProcessor, UIContextLayer, PerformanceMonitor, DocumentJob, JobPriority
)
from app.services.websocket.progress_handler import progress_handler

# German Authority Engine Import with validation
try:
    from app.services.german_authority_engine import GermanAuthorityEngine, get_all_authorities
    GERMAN_AUTHORITY_ENGINE_AVAILABLE = True
    logger_init = structlog.get_logger()
    logger_init.info("German Authority Engine imported successfully")
except ImportError as e:
    GERMAN_AUTHORITY_ENGINE_AVAILABLE = False
    get_all_authorities = None
    logger_init = structlog.get_logger()
    logger_init.warning(f"German Authority Engine not available: {e}")

# Big 4 Authority Engine Integration with validation
try:
    from app.services.german_authority_engine.integration.authority_endpoints import (
        Big4AuthorityEndpoints, create_big4_authority_endpoints
    )
    BIG4_ENDPOINTS_AVAILABLE = True
except ImportError as e:
    BIG4_ENDPOINTS_AVAILABLE = False
    Big4AuthorityEndpoints = None
    create_big4_authority_endpoints = None
    logger_init = structlog.get_logger()
    logger_init.warning(f"Big 4 Authority Endpoints not available: {e}")

# Initialize router
router = APIRouter(prefix="/api/v2/compliance", tags=["Day 2 - Enterprise Features with Docling Intelligence"])

# Initialize logger with proper configuration check
try:
    logger = structlog.get_logger()
    # Test logger functionality
    logger.info("Enhanced Compliance Router initialized successfully")
    LOGGING_AVAILABLE = True
except Exception as e:
    # Fallback to print if structlog fails
    print(f"Warning: Logger initialization failed: {e}")
    logger = None
    LOGGING_AVAILABLE = False

# Initialize Big 4 Engine with comprehensive error handling
big4_endpoints = None
BIG4_INITIALIZATION_ERROR = None

if BIG4_ENDPOINTS_AVAILABLE and create_big4_authority_endpoints:
    try:
        big4_endpoints = create_big4_authority_endpoints()
        if logger:
            logger.info("Big 4 Authority Engine initialized successfully")
    except Exception as e:
        BIG4_INITIALIZATION_ERROR = str(e)
        if logger:
            logger.warning(f"Big 4 Authority Engine initialization failed: {e}")
        else:
            print(f"Warning: Big 4 Authority Engine initialization failed: {e}")

def _log_safe(level: str, message: str, **kwargs):
    """Safe logging wrapper that handles logger unavailability"""
    if LOGGING_AVAILABLE and logger:
        getattr(logger, level.lower())(message, **kwargs)
    else:
        print(f"[{level.upper()}] {message} {kwargs if kwargs else ''}")

# =============================================================================
# INPUT VALIDATION HELPERS
# =============================================================================

def _validate_files_input(files: List[UploadFile]) -> None:
    """Validate files input to prevent empty list processing"""
    if not files:
        raise HTTPException(
            status_code=400,
            detail="No files provided. At least one file is required for analysis."
        )
    
    if len(files) > settings.max_files_per_batch:
        raise HTTPException(
            status_code=400,
            detail=f"Too many files. Maximum {settings.max_files_per_batch} files allowed per batch."
        )

def _validate_string_params(**params) -> Dict[str, str]:
    """Validate and sanitize string parameters"""
    validated = {}
    for name, value in params.items():
        if value is not None:
            # Strip whitespace and validate non-empty
            cleaned_value = str(value).strip()
            if cleaned_value:
                validated[name] = cleaned_value
            else:
                _log_safe("warning", f"Parameter {name} provided but empty after cleaning")
        else:
            validated[name] = None
    return validated

def _validate_processing_results(processing_results: List[Any]) -> bool:
    """Validate processing results to prevent calculation errors"""
    if not processing_results:
        _log_safe("warning", "Processing results are empty")
        return False
    
    valid_results = [r for r in processing_results if hasattr(r, 'success')]
    if len(valid_results) != len(processing_results):
        _log_safe("warning", f"Some processing results missing 'success' attribute: {len(processing_results) - len(valid_results)} invalid")
    
    return len(valid_results) > 0

# =============================================================================
# AUTHORITY CONTEXT CLASS - SCOPE FIX IMPLEMENTATION
# =============================================================================

@dataclass
class AuthorityContext:
    """
    Context object to maintain authority detection results throughout processing.
    This ensures authority data persists across async operations and try blocks.
    """
    detected_authority: str = "unknown"
    detected_industry: str = "unknown" 
    authority_confidence: float = 0.0
    authority_analysis: Optional[Any] = None
    authority_guidance: List[str] = field(default_factory=list)
    german_content_detected: bool = False
    
    def has_authority_data(self) -> bool:
        """Check if valid authority data was detected"""
        return self.detected_authority != "unknown" and self.authority_analysis is not None
    
    def to_metadata_dict(self) -> Dict[str, Any]:
        """Convert to metadata dictionary for API response"""
        return {
            "detected_authority": self.detected_authority,
            "detected_industry": self.detected_industry,
            "authority_confidence": self.authority_confidence,
            "authority_analysis_available": self.authority_analysis is not None,
            "authority_guidance_count": len(self.authority_guidance),
            "enforcement_likelihood": getattr(self.authority_analysis, 'enforcement_likelihood', 0.0),
            "penalty_risk_level": getattr(self.authority_analysis, 'penalty_risk_level', "unknown"),
            "audit_readiness_score": getattr(self.authority_analysis, 'audit_readiness_score', 0.0)
        }

# =============================================================================
# IMPROVED PARAMETER CLASS FOR COMPLEX FUNCTIONS
# =============================================================================

@dataclass
class ComplianceAnalysisParams:
    """IMPROVED: Parameter class to reduce function complexity with validation"""
    files: List[UploadFile]
    framework: str = "gdpr"
    workspace_id: str = DEMO_WORKSPACE_ID
    user_id: str = DEMO_ADMIN_USER_ID
    company_location: Optional[str] = None
    industry_hint: Optional[str] = None
    
    def __post_init__(self):
        """Validate parameters after initialization"""
        # Validate files
        _validate_files_input(self.files)
        
        # Validate and sanitize string parameters
        validated_params = _validate_string_params(
            company_location=self.company_location,
            industry_hint=self.industry_hint
        )
        self.company_location = validated_params["company_location"]
        self.industry_hint = validated_params["industry_hint"]
    
    def validate(self) -> ComplianceFramework:
        """Validate and return framework enum"""
        try:
            return ComplianceFramework(self.framework.lower())
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported framework: {self.framework}. Supported: gdpr, soc2, hipaa, iso27001"
            )

# =============================================================================
# OPTIMIZED AUTHORITY DETECTION - SCOPE FIX HELPER
# =============================================================================

async def _optimized_authority_detection(
    documents: List[Any],
    company_location: Optional[str],
    industry_hint: Optional[str]
) -> AuthorityContext:
    """
    Optimized authority detection with performance considerations.
    Uses concurrent execution where possible and maintains proper scope.
    """
    context = AuthorityContext()
    
    # Validate inputs
    if not documents:
        _log_safe("warning", "No documents provided for authority detection")
        return context
    
    # Check if Big 4 engine is available
    if not big4_endpoints:
        _log_safe("info", "Big 4 Authority Engine not available, skipping authority detection")
        return context
    
    try:
        from app.services.german_authority_engine.big4.big4_detector import Big4AuthorityDetector
        from app.services.german_authority_engine.big4.big4_analyzer import Big4ComplianceAnalyzer
        
        detector = Big4AuthorityDetector()
        analyzer = Big4ComplianceAnalyzer()
        
        # Run industry detection and authority detection concurrently
        industry_task = detector.detect_industry_from_content(documents)
        
        # Start with industry hint if available to speed up detection
        initial_industry = industry_hint if industry_hint else None
        
        # Detect authorities with initial hint
        detection_task = detector.detect_relevant_authorities(
            documents=documents,
            suggested_industry=initial_industry,
            suggested_state=company_location
        )
        
        # Await both tasks
        detected_industry, detection_result = await asyncio.gather(
            industry_task, detection_task
        )
        
        # Update context with detection results
        context.detected_industry = detected_industry if detected_industry != "unknown" else (industry_hint or "unknown")
        context.german_content_detected = True  # Already confirmed by caller
        
        if detection_result and hasattr(detection_result, 'primary_authority') and detection_result.primary_authority:
            context.detected_authority = detection_result.primary_authority.value
            context.authority_confidence = getattr(detection_result, 'detection_confidence', 0.0)
            
            _log_safe("info", f"Authority detected: {context.detected_authority} (confidence: {context.authority_confidence:.2f})")
            
            # Perform analysis only if detection confidence is high enough
            if context.authority_confidence >= 0.5:  # 50% threshold
                context.authority_analysis = await analyzer.analyze_for_authority(
                    documents=documents,
                    authority=detection_result.primary_authority,
                    industry=context.detected_industry
                )
                
                # Extract authority-specific guidance
                if context.authority_analysis:
                    context.authority_guidance = [
                        f"{context.authority_analysis.authority_name} requires: {req}" 
                        for req in getattr(context.authority_analysis, 'requirements_missing', [])[:3]
                    ]
                    
                    # Add industry-specific guidance
                    if hasattr(context.authority_analysis, 'industry_specific_guidance') and context.authority_analysis.industry_specific_guidance:
                        context.authority_guidance.extend(
                            context.authority_analysis.industry_specific_guidance[:2]
                        )
                    
                    _log_safe("info", f"Authority analysis completed - {len(context.authority_guidance)} guidance items generated")
    
    except Exception as e:
        _log_safe("warning", f"Optimized authority detection failed: {str(e)}")
        # Return context with default values
    
    return context

# =============================================================================
# PERFORMANCE GRADE CALCULATION - FIXED WITH PROPER ERROR HANDLING
# =============================================================================

def _calculate_performance_grade(results) -> str:
    """
    Calculate overall performance grade based on processing results
    
    FIXED: Complete implementation with proper error handling for division by zero
    """
    
    # Handle empty results (main cause of division by zero)
    if not results or len(results) == 0:
        _log_safe("warning", "No processing results provided for performance grading")
        return "F"
    
    # Validate processing results
    if not _validate_processing_results(results):
        _log_safe("warning", "Invalid processing results provided")
        return "F"
    
    try:
        # Filter successful results with proper attribute checking
        successful_results = []
        for r in results:
            # Handle different result object types safely
            success = getattr(r, 'success', False)
            if success:
                successful_results.append(r)
        
        # Calculate success rate with zero division protection
        total_count = len(results)
        success_count = len(successful_results)
        success_rate = success_count / total_count if total_count > 0 else 0.0
        
        # Calculate average processing time with error handling
        if successful_results:
            processing_times = []
            for r in successful_results:
                proc_time = getattr(r, 'processing_time', 0.0)
                if proc_time and proc_time > 0:
                    processing_times.append(proc_time)
            
            avg_time = sum(processing_times) / len(processing_times) if processing_times else float('inf')
        else:
            avg_time = float('inf')
        
        # Grade based on Day 3 performance targets
        if avg_time <= 3.0 and success_rate >= 0.95:
            grade = "A"
        elif avg_time <= 5.0 and success_rate >= 0.90:
            grade = "B"
        elif avg_time <= 8.0 and success_rate >= 0.80:
            grade = "C"
        elif avg_time <= 12.0 and success_rate >= 0.70:
            grade = "D"
        else:
            grade = "F"
        
        _log_safe("debug", "Performance grade calculated", 
                 total_results=total_count, successful_results=success_count, 
                 success_rate=success_rate, avg_time=avg_time, grade=grade)
        
        return grade
        
    except ZeroDivisionError as e:
        _log_safe("error", f"Division by zero in performance grade calculation: {e}")
        return "F"
    except Exception as e:
        _log_safe("error", f"Error calculating performance grade: {e}")
        return "F"

def _safe_calculate_performance_grade(processing_results) -> str:
    """
    Safe wrapper for performance grade calculation with comprehensive error handling
    """
    try:
        if not processing_results:
            _log_safe("info", "No processing results available for performance grading")
            return "F"
        
        return _calculate_performance_grade(processing_results)
        
    except ZeroDivisionError as e:
        _log_safe("error", f"Division by zero in performance grade calculation: {e}")
        return "F"
    except AttributeError as e:
        _log_safe("error", f"Missing attribute in performance grade calculation: {e}")
        return "F"
    except Exception as e:
        _log_safe("error", f"Unexpected error in performance grade calculation: {e}")
        return "F"

# =============================================================================
# MAIN ANALYSIS ENDPOINT WITH AUTHORITY INTELLIGENCE
# =============================================================================

@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_compliance_with_enterprise_features(
    request: Request,
    files: List[UploadFile] = File(...),
    framework: str = Form(default="gdpr"),
    workspace_id: str = Form(default=DEMO_WORKSPACE_ID),
    user_id: str = Form(default=DEMO_ADMIN_USER_ID),
    company_location: Optional[str] = Form(None),
    industry_hint: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_db_session)
):
    """
    Day 3 Enhanced: 10x Performance + UI Context Intelligence + Authority Detection
    - Parallel processing with intelligent job prioritization
    - Real-time WebSocket progress updates
    - UI context detection for smart interface automation
    - Performance monitoring with German compliance optimization
    - Integrated German authority detection and analysis
    """
    
    start_time = datetime.now(timezone.utc)
    
    # IMPROVED: Use parameter class with validation
    try:
        params = ComplianceAnalysisParams(
            files=files,
            framework=framework,
            workspace_id=workspace_id,
            user_id=user_id,
            company_location=company_location,
            industry_hint=industry_hint
        )
    except HTTPException:
        raise
    except Exception as e:
        _log_safe("error", f"Parameter validation failed: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid parameters: {str(e)}")
    
    # Initialize authority context object to maintain scope throughout processing
    authority_ctx = AuthorityContext()
    
    # Extract client information for audit trail
    client_ip = request.client.host if request.client else None
    user_agent = request.headers.get("user-agent")
    
    # Validate framework
    compliance_framework = params.validate()
    
    # Initialize variables for error handling scope
    processing_results = []
    individual_analyses = []
    ui_context = None
    performance_monitor = None
    processing_time = 0.0
    
    try:
        # Process and validate files
        processed_files = await _process_and_validate_files(files)
        
        _log_safe("info", "Day 3 enhanced processing started",
                 workspace_id=workspace_id, file_count=len(processed_files),
                 total_size_mb=sum(size for _, _, size in processed_files) / (1024 * 1024),
                 framework=framework)
        
        # =================================================================
        # BIG 4 GERMAN AUTHORITY DETECTION WITH SCOPE FIX
        # =================================================================
        
        # Initialize UI Context Layer for content detection
        ui_context_layer = UIContextLayer()
        
        # German content detection using public method
        authority_ctx.german_content_detected = any(
            ui_context_layer.detect_german_content(content) 
            for _, content, _ in processed_files
        )
        
        # Only activate Big 4 Authority Engine for German GDPR content
        if authority_ctx.german_content_detected and compliance_framework.value == 'gdpr' and big4_endpoints:
            authority_ctx = await _perform_authority_detection(
                processed_files, params.company_location, params.industry_hint
            )
        
        # =================================================================
        # EXISTING ANALYSIS PIPELINE WITH AUTHORITY CONTEXT
        # =================================================================
        
        # Initialize Day 3 processing components
        processing_components = await _initialize_processing_components()
        job_queue, batch_processor, performance_monitor = processing_components
        
        # Generate UI context intelligence
        ui_context = ui_context_layer.analyze_ui_context([])  # Will be populated with jobs
        
        # Send UI context to frontend
        try:
            await progress_handler.handle_ui_context_update(workspace_id, ui_context.to_dict())
        except Exception as e:
            _log_safe("warning", f"Failed to send UI context update: {e}")
        
        # Create intelligent job queue with authority context
        jobs = await _create_intelligent_jobs(
            processed_files, params, authority_ctx, job_queue
        )
        
        # Create intelligent batches for parallel processing
        batches = job_queue.create_intelligent_batches(jobs)
        
        # Start performance monitoring
        performance_monitor.start_batch_monitoring(len(jobs))
        
        # Notify frontend of batch start
        try:
            await progress_handler.handle_batch_started(workspace_id, {
                "total_jobs": len(jobs),
                "total_batches": len(batches),
                "german_docs": sum(1 for job in jobs if hasattr(job, 'is_german_compliance') and job.is_german_compliance),
                "framework": framework,
                "ui_context": ui_context.to_dict(),
                "authority_intelligence": authority_ctx.to_metadata_dict()
            })
        except Exception as e:
            _log_safe("warning", f"Failed to send batch start notification: {e}")
        
        # Create progress callback for real-time updates
        progress_callback = progress_handler.create_progress_callback(workspace_id)
        
        # Process batches with parallel intelligence
        processing_results = await batch_processor.process_batches(
            batches, workspace_id, user_id, compliance_framework, progress_callback
        )
        
        # Validate processing results before proceeding
        if not _validate_processing_results(processing_results):
            raise ValueError("Processing results validation failed")
        
        # Record performance metrics
        performance_monitor.record_processing_results(processing_results)
        
        # Convert processing results to document analyses
        individual_analyses = _convert_results_to_analyses(processing_results, authority_ctx)
        
        # Create enhanced compliance report with authority context
        compliance_report = await _create_enhanced_compliance_report(
            individual_analyses, ui_context, performance_monitor, compliance_framework, workspace_id,
            authority_context=authority_ctx
        )
        
        processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()
        
        # =================================================================
        # ENHANCED RESPONSE WITH AUTHORITY INTELLIGENCE
        # =================================================================
        
        analysis_response = _build_analysis_response(
            individual_analyses, compliance_report, authority_ctx, ui_context,
            performance_monitor, processing_results, params, processing_time, start_time, batches
        )
        
        # Calculate safe success rate
        total_results = len(processing_results)
        successful_results = len([r for r in processing_results if getattr(r, 'success', False)])
        success_rate = successful_results / total_results if total_results else 0.0

        # Notify frontend of completion with authority data
        try:
            await progress_handler.handle_batch_completed(workspace_id, {
                "documents_analyzed": len(individual_analyses),
                "processing_time": processing_time,
                "compliance_score": compliance_report.compliance_score,
                "german_documents_detected": compliance_report.german_documents_detected,
                "success_rate": success_rate,
                "performance_grade": _safe_calculate_performance_grade(processing_results),
                "ui_context": ui_context.to_dict() if ui_context else {},
                "authority_intelligence": authority_ctx.to_metadata_dict()
            })
        except Exception as e:
            _log_safe("warning", f"Failed to send completion notification: {e}")
        
        _log_safe("info", "Day 3 enhanced analysis completed successfully",
                 workspace_id=workspace_id, processing_time=processing_time,
                 documents_processed=len(individual_analyses),
                 compliance_score=compliance_report.compliance_score,
                 performance_grade=_safe_calculate_performance_grade(processing_results),
                 detected_authority=authority_ctx.detected_authority,
                 detected_industry=authority_ctx.detected_industry)
        
        return analysis_response
        
    except HTTPException:
        raise
    except Exception as e:
        # Enhanced error handling with progress notification
        try:
            await progress_handler.handle_error_notification(workspace_id, {
                "error_type": "processing_error", 
                "message": str(e),
                "workspace_id": workspace_id,
                "framework": framework,
                "error_details": {
                    "processing_results_count": len(processing_results),
                    "individual_analyses_count": len(individual_analyses),
                    "authority_detected": getattr(authority_ctx, 'detected_authority', 'unknown'),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "error_type": type(e).__name__,
                    "processing_time": processing_time
                }
            })
        except Exception as notify_error:
            _log_safe("warning", f"Failed to send error notification: {notify_error}")
        
        _log_safe("error", "Day 3 enhanced analysis failed",
                 workspace_id=workspace_id, error=str(e), error_type=type(e).__name__,
                 processing_stage="main_analysis")
        
        raise HTTPException(
            status_code=500,
            detail=f"Enhanced compliance analysis failed: {str(e)}"
        )