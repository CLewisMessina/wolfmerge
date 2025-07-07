# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.routers import compliance

# Create FastAPI app
app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    description="AI-powered compliance document analysis for German enterprises"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(compliance.router)

@app.get("/")
async def root():
    return {
        "message": "WolfMerge Compliance API",
        "version": settings.api_version,
        "features": ["German DSGVO analysis", "Multi-framework support", "GDPR-compliant processing"],
        "day": "Day 1 - Compliance Foundation"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "german_support": True,
        "frameworks": ["gdpr", "soc2", "hipaa", "iso27001"]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug
    )