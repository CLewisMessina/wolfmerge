# app/main.py
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import structlog
import traceback
import uuid
import json
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from openai import OpenAI

from app.config import settings
from app.routers import compliance  # Day 1 router (backward compatibility)
from app.routers import enhanced_compliance  # Day 2 enterprise router
from app.database import create_tables, init_demo_data, health_check
from app.utils.smart_docling import get_docling_status  # NEW: Added for Smart Docling check
from app.services.websocket.manager import websocket_manager

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management for Day 2 enterprise features"""
    
    # Startup
    logger.info("Starting WolfMerge Enterprise Compliance Platform v2.0")
    
    try:
        # Initialize database tables
        logger.info("Creating database tables...")
        await create_tables()
        logger.info("Database tables created successfully")
        
        # Initialize demo data for Day 2 testing
        logger.info("Initializing demo data...")
        await init_demo_data()
        logger.info("Demo data initialized successfully")
        
        # Verify database health
        db_health = await health_check()
        if db_health.get("database_connected"):
            logger.info(
                "Database health check passed",
                tables_created=db_health.get("tables_created"),
                demo_data_loaded=db_health.get("demo_data_loaded"),
                eu_region=db_health.get("eu_region")
            )
        else:
            logger.error("Database health check failed")
        
        # Test OpenAI connectivity on startup
        try:
            client = OpenAI(api_key=settings.openai_api_key)
            logger.info("OpenAI client initialized successfully")
        except Exception as e:
            logger.error("OpenAI initialization failed", error=str(e))
        
        # Log startup completion - FIXED: Remove problematic kwargs
        logger.info("WolfMerge Enterprise Platform started successfully")
        
    except Exception as e:
        logger.error("Startup failed", error=str(e))
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down WolfMerge Enterprise Platform")
    
    try:
        logger.info("Cleanup completed successfully")
    except Exception as e:
        logger.error("Shutdown cleanup failed", error=str(e))

# Create FastAPI app with enhanced enterprise features
app = FastAPI(
    title="WolfMerge Enterprise Compliance Platform",
    version="2.0.0",
    description="""
    AI-powered compliance analysis platform for German enterprises.
    
    **Day 2 Enterprise Features:**
    - Docling intelligent document processing with semantic chunking
    - Team workspace collaboration with PostgreSQL backend
    - Advanced German DSGVO compliance analysis
    - EU cloud deployment with GDPR-compliant audit trails
    - Chunk-level compliance insights for detailed audit preparation
    
    **German Market Focus:**
    - Native DSGVO terminology recognition and article mapping
    - German supervisory authority compliance verification  
    - Industry-specific templates for automotive, healthcare, manufacturing
    - SME-focused workflows at enterprise-grade quality
    
    **API Versions:**
    - `/api/compliance` - Day 1 basic analysis (backward compatible)
    - `/api/v2/compliance` - Day 2 enterprise features with Docling intelligence
    """,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Security middleware for enterprise deployment
app.add_middleware(
    TrustedHostMiddleware, 
    allowed_hosts=[
        "localhost",
        "127.0.0.1", 
        "api.wolfmerge.com",
        "dev-api.wolfmerge.com",
        "*.railway.app"
    ]
)

# Enhanced CORS middleware for enterprise deployment
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins + [
        "https://app.wolfmerge.com",
        "https://dev-app.wolfmerge.com",
        "https://*.wolfmerge.com"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["X-Analysis-ID", "X-Workspace-ID", "X-Processing-Time"]
)

# Request logging middleware for audit trails
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests for audit trails and monitoring"""
    
    start_time = datetime.now(timezone.utc)
    request_id = str(uuid.uuid4())
    
    # Log request start
    logger.info(
        "Request started",
        request_id=request_id,
        method=request.method,
        url=str(request.url),
        client_ip=request.client.host if request.client else None,
        user_agent=request.headers.get("user-agent"),
        content_length=request.headers.get("content-length")
    )
    
    try:
        response = await call_next(request)
        
        processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()
        
        # Add processing time and request ID headers
        response.headers["X-Processing-Time"] = str(processing_time)
        response.headers["X-Request-ID"] = request_id
        
        # Log successful request
        logger.info(
            "Request completed",
            request_id=request_id,
            method=request.method,
            url=str(request.url),
            status_code=response.status_code,
            processing_time=processing_time
        )
        
        return response
        
    except Exception as e:
        processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()
        
        # Log failed request
        logger.error(
            "Request failed",
            request_id=request_id,
            method=request.method,
            url=str(request.url),
            error=str(e),
            processing_time=processing_time
        )
        
        # Return structured error response
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "message": "Request processing failed",
                "request_id": request_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "processing_time": processing_time
            },
            headers={
                "X-Request-ID": request_id,
                "X-Processing-Time": str(processing_time)
            }
        )

# Include API routers with backward compatibility
app.include_router(
    compliance.router, 
    tags=["Day 1 - Basic Compliance (Backward Compatible)"]
)

app.include_router(
    enhanced_compliance.router, 
    tags=["Day 2 - Enterprise Features with Docling Intelligence"]
)

@app.websocket("/ws/{workspace_id}")
async def websocket_endpoint(websocket: WebSocket, workspace_id: str):
    """
    Day 3: Real-time progress tracking WebSocket endpoint
    
    Provides live updates for:
    - Document processing progress
    - Batch status updates
    - UI context intelligence
    - Performance metrics
    - Error notifications
    """
    
    connection_id = None
    
    try:
        # Establish WebSocket connection
        connection_id = await websocket_manager.connect(websocket, workspace_id)
        
        logger.info(
            "WebSocket connection established",
            workspace_id=workspace_id,
            connection_id=connection_id
        )
        
        # Keep connection alive and handle messages
        while True:
            try:
                # Wait for client messages (ping/pong, etc.)
                data = await websocket.receive_text()
                
                # Handle client messages if needed
                try:
                    message = json.loads(data)
                    if message.get("type") == "ping":
                        await websocket.send_text(json.dumps({
                            "type": "pong",
                            "timestamp": datetime.now(timezone.utc).isoformat()
                        }))
                except json.JSONDecodeError:
                    pass  # Ignore invalid JSON
                    
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.warning(
                    "WebSocket message handling error",
                    workspace_id=workspace_id,
                    connection_id=connection_id,
                    error=str(e)
                )
                break
                
    except Exception as e:
        logger.error(
            "WebSocket connection error",
            workspace_id=workspace_id,
            error=str(e)
        )
    finally:
        # Clean up connection
        if connection_id:
            websocket_manager.disconnect(workspace_id, connection_id)
            
            logger.info(
                "WebSocket connection closed",
                workspace_id=workspace_id,
                connection_id=connection_id
            )

# Comprehensive exception handler
@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle all unhandled exceptions"""
    request_id = str(uuid.uuid4())
    
    logger.error(
        "Unhandled exception",
        request_id=request_id,
        path=request.url.path,
        error=str(exc),
        error_type=type(exc).__name__
    )
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred",
            "request_id": request_id,
            "timestamp": datetime.now(timezone.utc).isoformat()
        },
        headers={
            "X-Request-ID": request_id
        }
    )

@app.get("/")
async def root():
    """Root endpoint with comprehensive platform information"""
    
    return {
        "message": "WolfMerge Enterprise Compliance Platform",
        "version": "2.0.0",
        "day": "Day 2 - Enterprise Cloud Platform with Docling Intelligence",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "description": "AI-powered German DSGVO compliance analysis for SME and enterprise teams",
        
        "features": {
            "day_1_features": [
                "German DSGVO compliance analysis",
                "Multi-framework support (GDPR/SOC2/HIPAA/ISO27001)",
                "Professional API with OpenAPI documentation",
                "EU cloud deployment"
            ],
            "day_2_enhancements": [
                "Docling intelligent document processing",
                "Team workspace collaboration",
                "Chunk-level compliance analysis",
                "GDPR-compliant audit trails",
                "German industry-specific templates",
                "Enterprise-grade security and monitoring"
            ]
        },
        
        "api_endpoints": {
            "day_1_basic": {
                "analyze": "/api/compliance/analyze",
                "frameworks": "/api/compliance/frameworks",
                "health": "/health"
            },
            "day_2_enterprise": {
                "analyze": "/api/v2/compliance/analyze",
                "history": "/api/v2/compliance/workspace/{workspace_id}/history",
                "audit_trail": "/api/v2/compliance/workspace/{workspace_id}/audit-trail",
                "compliance_report": "/api/v2/compliance/workspace/{workspace_id}/compliance-report",
                "german_templates": "/api/v2/compliance/templates/german-industry",
                "health": "/api/v2/compliance/health"
            }
        },
        
        "german_compliance": {
            "dsgvo_articles_supported": ["Art. 5", "Art. 6", "Art. 7", "Art. 13-18", "Art. 20", "Art. 25", "Art. 30", "Art. 32", "Art. 35"],
            "german_authorities": ["BfDI", "BayLDA", "LfDI"],
            "industry_templates": ["automotive", "healthcare", "manufacturing"],
            "language_support": ["German (de)", "English (en)"],
            "legal_framework_coverage": ["DSGVO", "BDSG", "GDPR"]
        },
        
        "enterprise_ready": {
            "eu_cloud_deployment": settings.eu_region,
            "gdpr_compliance": settings.gdpr_compliance,
            "audit_trails": settings.audit_logging,
            "team_workspaces": settings.enable_workspaces,
            "document_intelligence": settings.docling_enabled,
            "data_residency": settings.data_residency
        },
        
        "target_market": {
            "primary": "German SMEs (10-500 employees)",
            "secondary": "German compliance consultants", 
            "pricing_model": "€200/month SME tier vs €2000+/month enterprise alternatives",
            "value_proposition": "Enterprise-grade German DSGVO compliance at SME-friendly pricing"
        },
        
        "documentation": {
            "interactive_docs": "/docs",
            "redoc": "/redoc",
            "openapi_spec": "/openapi.json"
        }
    }

@app.get("/health")
async def health_check_endpoint():
    """Comprehensive health check for Day 2 enterprise platform"""
    
    try:
        # Get database health
        db_health = await health_check()
        
        # Enhanced health checks for all services
        service_checks = {
            "api_server": True,  # Always true if we get here
            "database": False,
            "openai": False,
            "docling": False,
            "smart_docling": False  # NEW: Added Smart Docling check
        }
        
        # Check database
        service_checks["database"] = db_health.get("database_connected", False)
        
        # Check OpenAI
        try:
            client = OpenAI(api_key=settings.openai_api_key)
            # Simple check - just see if client initializes
            service_checks["openai"] = True
        except Exception as e:
            logger.warning("OpenAI health check failed", error=str(e))
        
        # Check Docling (simple check - see if it's enabled)
        service_checks["docling"] = settings.docling_enabled
        
        # NEW: Check Smart Docling
        docling_status = get_docling_status()
        service_checks["smart_docling"] = docling_status["docling_available"]
        
        # Overall system health
        all_healthy = all([
            service_checks["database"],
            service_checks["openai"],
            # Docling is optional, so don't require it for overall health
        ])
        
        health_status = {
            "status": "healthy" if all_healthy else "degraded",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "version": "2.0.0",
            "environment": settings.environment,
            
            "core_services": service_checks,
            
            "database_details": db_health,
            
            "smart_docling_status": docling_status,  # NEW: Added Smart Docling status
            
            "websocket_status": {
                "manager_active": True,
                "active_connections": websocket_manager.active_connections,
                "active_workspaces": len(websocket_manager.connections)
            },
            
            "compliance_features": {
                "german_dsgvo_analysis": "active",
                "multi_framework_support": "active", 
                "audit_trails": "active" if settings.audit_logging else "inactive",
                "eu_data_residency": "compliant" if settings.eu_region else "non_compliant",
                "gdpr_compliance": "enforced" if settings.gdpr_compliance else "disabled"
            },
            
            "performance_metrics": {
                "max_file_size_mb": settings.max_file_size_mb,
                "max_files_per_batch": settings.max_files_per_batch,
                "max_chunks_per_document": settings.max_chunks_per_document,
                "data_retention_hours": settings.data_retention_hours,
                "audit_retention_days": settings.audit_retention_days
            },
            
            "day_2_capabilities": {
                "docling_intelligence": settings.docling_enabled,
                "team_workspaces": settings.enable_workspaces,
                "chunk_level_analysis": True,
                "german_industry_templates": True,
                "enterprise_audit_reports": True
            }
        }
        
        logger.info("Health check completed", status=health_status["status"])
        
        return health_status
        
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        
        return {
            "status": "unhealthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "error": str(e),
            "version": "2.0.0"
        }

@app.get("/api/status")
async def api_status():
    """Detailed API status for monitoring and debugging"""
    
    return {
        "api_version": "2.0.0",
        "service_name": "WolfMerge Enterprise Compliance Platform",
        "deployment_info": {
            "environment": settings.environment,
            "eu_region": settings.eu_region,
            "debug_mode": settings.debug,
            "data_residency": settings.data_residency
        },
        
        "feature_flags": {
            "docling_enabled": settings.docling_enabled,
            "workspaces_enabled": settings.enable_workspaces,
            "audit_logging": settings.audit_logging,
            "german_templates": True,
            "chunk_analysis": True
        },
        
        "api_capabilities": {
            "backward_compatible": True,
            "day_1_endpoints": "active",
            "day_2_endpoints": "active",
            "german_compliance": "specialized",
            "enterprise_features": "full"
        },
        
        "supported_formats": settings.allowed_extensions,
        "supported_frameworks": ["gdpr", "soc2", "hipaa", "iso27001"],
        "supported_languages": ["de", "en"],
        
        "limits": {
            "max_file_size_mb": settings.max_file_size_mb,
            "max_files_per_batch": settings.max_files_per_batch,
            "max_workspace_size_mb": settings.max_workspace_size_mb,
            "max_users_per_workspace": settings.max_users_per_workspace
        }
    }

@app.get("/api/v2/websocket/stats")
async def get_websocket_stats():
    """Get WebSocket connection statistics"""
    
    return {
        "websocket_stats": websocket_manager.get_manager_stats(),
        "day3_features": {
            "real_time_progress": True,
            "ui_context_updates": True,
            "performance_monitoring": True,
            "error_notifications": True
        },
        "connection_info": {
            "endpoint": "/ws/{workspace_id}",
            "supported_messages": [
                "job_progress",
                "batch_started", 
                "batch_completed",
                "ui_context_update",
                "performance_update",
                "error_notification"
            ]
        }
    }

# Exception handlers for enterprise-grade error handling
@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    """Custom 404 handler with helpful information"""
    
    request_id = str(uuid.uuid4())
    
    logger.warning(
        "404 Not Found",
        request_id=request_id,
        path=request.url.path,
        method=request.method,
        client_ip=request.client.host if request.client else None
    )
    
    return JSONResponse(
        status_code=404,
        content={
            "error": "Not Found",
            "message": f"The requested endpoint {request.url.path} was not found",
            "available_endpoints": {
                "day_1_api": "/api/compliance/*",
                "day_2_api": "/api/v2/compliance/*",
                "documentation": "/docs",
                "health_check": "/health"
            },
            "request_id": request_id,
            "timestamp": datetime.now(timezone.utc).isoformat()
        },
        headers={
            "X-Request-ID": request_id
        }
    )

@app.exception_handler(500)
async def internal_error_handler(request: Request, exc):
    """Custom 500 handler with enterprise error tracking"""
    
    request_id = str(uuid.uuid4())
    
    logger.error(
        "500 Internal Server Error",
        request_id=request_id,
        path=request.url.path,
        method=request.method,
        error=str(exc),
        client_ip=request.client.host if request.client else None
    )
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred while processing your request",
            "support_info": {
                "contact": "For support, please check the health endpoint: /health",
                "documentation": "/docs",
                "request_id": request_id,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        },
        headers={
            "X-Request-ID": request_id
        }
    )