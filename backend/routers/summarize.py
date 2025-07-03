# backend/routers/summarize.py

from fastapi import APIRouter, File, UploadFile, HTTPException
from typing import List
import io
from services.openai_service import OpenAIService

router = APIRouter()

# Removed global instantiation - will create service instance in endpoint instead

@router.post("/summarize")
async def summarize_documents(files: List[UploadFile] = File(...)):
    # Create service instance here to ensure .env is loaded first
    try:
        openai_service = OpenAIService()
    except ValueError as e:
        raise HTTPException(
            status_code=500, 
            detail=f"OpenAI service initialization failed: {str(e)}"
        )
    
    if len(files) > 3:
        raise HTTPException(
            status_code=400, 
            detail="Maximum 3 files allowed for Day 1 demo"
        )
    
    try:
        # Extract text from files
        documents = []
        for file in files:
            if not file.filename.endswith(('.txt', '.md')):
                raise HTTPException(
                    status_code=400, 
                    detail=f"Unsupported file type: {file.filename}. Only .txt and .md files supported on Day 1."
                )
            
            content = await file.read()
            text = content.decode('utf-8')
            
            # Basic validation
            if len(text.strip()) < 10:
                raise HTTPException(
                    status_code=400,
                    detail=f"File {file.filename} appears to be empty or too short"
                )
            
            documents.append({
                "filename": file.filename,
                "content": text,
                "size": len(text)
            })
        
        # Generate individual summaries
        individual_summaries = []
        for doc in documents:
            summary = await openai_service.summarize_document(
                doc["content"], 
                doc["filename"]
            )
            individual_summaries.append({
                "filename": doc["filename"],
                "summary": summary,
                "original_size": doc["size"]
            })
        
        # Generate unified summary
        unified_summary = await openai_service.create_unified_summary(individual_summaries)
        
        return {
            "individual_summaries": individual_summaries,
            "unified_summary": unified_summary,
            "document_count": len(documents),
            "total_size": sum(doc["size"] for doc in documents),
            "processing_time": "~2 minutes saved"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Unexpected error: {str(e)}")  # For debugging
        raise HTTPException(status_code=500, detail="Analysis failed. Please try again.")