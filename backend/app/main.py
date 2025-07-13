# app/main.py - Updated for Refactored Compliance Structure
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
# UPDATED: Import refactored compliance router instead of old ones
from app.routers.compliance import compliance_router
from app.database import create_tables, init_demo_data, health_check
from app.utils.smart_docling import get_docling_status
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
    """Application lifespan management for enterprise features"""
    
    # Startup
    logger.info("Starting WolfMerge Enterprise Compliance Platform v2.0 - Refactored")
    
    try:
        # Initialize database tables
        logger.info("Creating database tables...")
        await create_tables()
        logger.info("Database tables created successfully")
        
        # Initialize demo data
        logger.info("Initializing demo data...")
        await init_demo_data()
        logger.info("Demo data initialized successfully")
        
        # Verify database health
        db_health = await health_check()
        if db_health.get("database_connected"):
            logger.info(
                "Database health check passed",
                tables_created=db_health.get("tables_created"),
                demo_data_loaded=db_health.get("demo_data_loaded")
            )
        
        # Check Docling status
        docling_status = get_docling_status()
        logger.info(
            "Docling status checked",
            available=docling_status.get("docling_available"),
            environment=docling_status.get("environment")
        )
        
        # Verify OpenAI connection
        try:
            client = OpenAI(api_key=settings.openai_api_key)
            models = client.models.list()
            logger.info("OpenAI connection verified successfully")
        except Exception as e:
            logger.warning(f"OpenAI connection check failed: {e}")
        
        # Log refactored compliance system status
        logger.info(
            "Refactored compliance system initialized",
            max_file_size_mb=settings.max_file_size_mb,
            max_files_per_batch=settings.max_files_per_batch,
            authority_engine_enabled=settings.big4_authority_engine_enabled,
            frameworks_supported=["gdpr", "soc2", "hipaa", "iso27001"]
        )
        
        yield
        
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise
    
    # Shutdown
    logger.info("Shutting down WolfMerge Enterprise Compliance Platform")

# Create FastAPI app with lifespan management
app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    description="""
    🐺 **WolfMerge Enterprise Compliance Platform v2.0 - Refactored Architecture**
    
    Advanced compliance analysis with German authority intelligence, powered by a clean, maintainable architecture.
    
    ## 🏗️ **Refactored Architecture Benefits**
    - **Modular Services**: Clean separation of concerns
    - **Enhanced Debugging**: Isolated error handling per service  
    - **Performance Monitoring**: Real-time analytics and optimization
    - **Authority Intelligence**: German compliance specialization
    - **Scalable Design**: Easy to extend and maintain
    
    ## 🚀 **Core Features**
    - **Multi-Framework Analysis**: GDPR/DSGVO, SOC 2, HIPAA, ISO 27001
    - **German Authority Detection**: BfDI, BayLDA, LfD automatic identification
    - **Parallel Processing**: Intelligent job queue with priority handling
    - **Real-time Progress**: WebSocket updates during processing
    - **Team Workspaces**: Collaborative compliance management
    - **EU Cloud Deployment**: GDPR-compliant data residency
    
    ## 🏛️ **German Authority Intelligence**
    - **BfDI**: Federal Commissioner (international transfers)
    - **BayLDA**: Bavarian State Office (automotive focus)  
    - **LfD BW**: Baden-Württemberg Commissioner (manufacturing)
    - **Industry-Specific**: Automotive, healthcare, manufacturing
    """,
    lifespan=lifespan,
    debug=settings.debug
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add trusted host middleware for security
if settings.is_production:
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["*.wolfmerge.com", "*.railway.app", "localhost"]
    )

# Global exception handler for better error responses
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler with request tracking"""
    
    request_id = str(uuid.uuid4())
    processing_time = 0.0
    
    try:
        # Calculate processing time if available
        if hasattr(request.state, "start_time"):
            processing_time = (datetime.now(timezone.utc) - request.state.start_time).total_seconds()
        
        # Log the error with context
        logger.error(
            "Global exception handler triggered",
            request_id=request_id,
            method=request.method,
            url=str(request.url),
            error_type=type(exc).__name__,
            error_message=str(exc),
            processing_time=processing_time,
            exc_info=True
        )
        
    except Exception as logging_error:
        logger.error(f"Failed to log exception: {logging_error}")
    
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

# UPDATED: Include refactored compliance router
app.include_router(
    compliance_router,
    tags=["Compliance Analysis - Refactored v2.0"]
)

@app.websocket("/ws/{workspace_id}")
async def websocket_endpoint(websocket: WebSocket, workspace_id: str):
    """
    Real-time progress tracking WebSocket endpoint
    
    Provides live updates for:
    - Document processing progress
    - Batch status updates
    - Performance metrics
    - Error notifications
    - Authority detection results
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
                
                # Handle client messages
                try:
                    message = json.loads(data)
                    message_type = message.get("type")
                    
                    if message_type == "ping":
                        await websocket.send_text(json.dumps({
                            "type": "pong",
                            "timestamp": datetime.now(timezone.utc).isoformat()
                        }))
                    elif message_type == "subscribe_progress":
                        # Client wants to subscribe to progress updates
                        session_id = message.get("session_id")
                        if session_id:
                            await websocket_manager.subscribe_to_session(
                                connection_id, session_id
                            )
                    
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON message from client: {data}")
                
            except WebSocketDisconnect:
                logger.info(f"WebSocket client disconnected: {workspace_id}")
                break
            except Exception as e:
                logger.error(f"WebSocket message handling error: {e}")
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": "Message processing failed"
                }))
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket connection closed: {workspace_id}")
    except Exception as e:
        logger.error(f"WebSocket connection error: {e}")
    finally:
        if connection_id:
            await websocket_manager.disconnect(connection_id)

@app.get("/")
async def root():
    """Root endpoint with refactored system information"""
    return {
        "message": "🐺 WolfMerge Enterprise Compliance Platform",
        "version": settings.api_version,
        "architecture": "Refactored v2.0",
        "status": "operational",
        "features": {
            "compliance_analysis": "Multi-framework support",
            "german_authorities": "BfDI, BayLDA, LfD detection",
            "parallel_processing": "Intelligent job queue",
            "real_time_progress": "WebSocket updates",
            "team_workspaces": "Collaborative compliance",
            "refactored_architecture": "Clean, maintainable services"
        },
        "endpoints": {
            "compliance_analysis": "/api/v2/compliance/analyze",
            "health_check": "/api/v2/compliance/health",
            "performance_summary": "/api/v2/compliance/performance/summary",
            "supported_frameworks": "/api/v2/compliance/frameworks",
            "german_authorities": "/api/v2/compliance/authorities/german",
            "websocket": "/ws/{workspace_id}"
        },
        "documentation": "/docs",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

@app.get("/health")
async def health_check_endpoint():
    """System health check endpoint"""
    
    try:
        # Check database
        db_health = await health_check()
        
        # Check Docling
        docling_status = get_docling_status()
        
        # Check settings
        settings_valid = all([
            settings.openai_api_key,
            settings.database_url,
            settings.secret_key
        ])
        
        health_data = {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "version": settings.api_version,
            "architecture": "refactored_v2.0",
            "components": {
                "database": {
                    "status": "connected" if db_health.get("database_connected") else "disconnected",
                    "tables_created": db_health.get("tables_created", False)
                },
                "docling": {
                    "status": "available" if docling_status.get("docling_available") else "unavailable",
                    "environment": docling_status.get("environment")
                },
                "settings": {
                    "status": "valid" if settings_valid else "invalid",
                    "openai_configured": bool(settings.openai_api_key),
                    "authority_engine_enabled": settings.big4_authority_engine_enabled
                }
            }
        }
        
        # Determine overall status
        component_statuses = [
            health_data["components"]["database"]["status"] == "connected",
            health_data["components"]["settings"]["status"] == "valid"
        ]
        
        if not all(component_statuses):
            health_data["status"] = "degraded"
        
        return health_data
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }