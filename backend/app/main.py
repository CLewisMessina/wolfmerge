# app/main.py - Debug Version to Isolate Import Issue
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.routers import compliance  # Day 1 router (working)

# TEMPORARILY COMMENT OUT Day 2 router to isolate import issue
# from app.routers import enhanced_compliance  # Day 2 enterprise router

# TEMPORARILY COMMENT OUT database imports to isolate issue
# from app.database import create_tables, init_demo_data, health_check

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
    """Simplified lifespan for debugging"""
    
    # Startup
    logger.info("Starting WolfMerge Platform - Debug Mode")
    
    try:
        # TEMPORARILY SKIP database initialization to isolate import issue
        # await create_tables()
        # await init_demo_data()
        
        logger.info("WolfMerge Platform started successfully - Debug Mode")
        
    except Exception as e:
        logger.error("Startup failed", error=str(e))
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down WolfMerge Platform")

# Create simplified FastAPI app for debugging
app = FastAPI(
    title="WolfMerge Compliance Platform - Debug",
    version="2.0.0-debug",
    description="Debug version to isolate import issues",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Basic middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Simplified for debugging
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Include only Day 1 router (working)
app.include_router(
    compliance.router, 
    tags=["Day 1 - Basic Compliance (Working)"]
)

# TEMPORARILY COMMENT OUT Day 2 router
# app.include_router(
#     enhanced_compliance.router, 
#     tags=["Day 2 - Enterprise Features"]
# )

@app.get("/")
async def root():
    """Simplified root endpoint for debugging"""
    
    return {
        "message": "WolfMerge Compliance Platform - Debug Mode",
        "version": "2.0.0-debug",
        "status": "Day 1 working, Day 2 temporarily disabled for debugging",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        
        "working_endpoints": {
            "day_1_basic": {
                "analyze": "/api/compliance/analyze",
                "frameworks": "/api/compliance/frameworks",
                "health": "/health"
            }
        },
        
        "debug_info": {
            "day_2_router": "temporarily disabled",
            "database_init": "temporarily disabled",
            "purpose": "isolate import issues"
        }
    }

@app.get("/health")
async def simple_health_check():
    """Simplified health check for debugging"""
    
    return {
        "status": "healthy",
        "version": "2.0.0-debug",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "debug_mode": True,
        "day_1_features": "active",
        "day_2_features": "disabled_for_debugging"
    }

@app.get("/debug")
async def debug_info():
    """Debug endpoint to check what's working"""
    
    try:
        # Test basic imports
        import openai
        openai_available = True
    except ImportError:
        openai_available = False
    
    try:
        # Test if we can import our basic models
        from app.models.compliance import ComplianceFramework
        compliance_models_available = True
    except ImportError:
        compliance_models_available = False
    
    try:
        # Test if we can import German detection
        from app.utils.german_detection import GermanComplianceDetector
        german_detection_available = True
    except ImportError:
        german_detection_available = False
    
    return {
        "import_status": {
            "openai": openai_available,
            "compliance_models": compliance_models_available,
            "german_detection": german_detection_available
        },
        "environment": {
            "docling_enabled": getattr(settings, 'docling_enabled', False),
            "eu_region": getattr(settings, 'eu_region', False),
            "debug": getattr(settings, 'debug', False)
        }
    }

if __name__ == "__main__":
    import uvicorn
    
    logger.info("Starting WolfMerge Debug Mode")
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Enable reload for debugging
        log_level="debug"
    )