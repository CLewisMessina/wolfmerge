# backend/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import summarize
import os
from dotenv import load_dotenv

load_dotenv()
print(f"API Key loaded: {os.getenv('OPENAI_API_KEY')[:10]}...")  # Shows first 10 chars

app = FastAPI(title="WolfMerge API", version="1.0.0")


# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(summarize.router, prefix="/api")

@app.get("/")
async def root():
    return {"message": "WolfMerge API is running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)