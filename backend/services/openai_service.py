# backend/services/openai_service.py
from openai import OpenAI
import os
from typing import List, Dict

class OpenAIService:
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        self.client = OpenAI(api_key=api_key)
        self.model = "gpt-4o-mini"  # Cost-effective choice
        self.max_tokens_individual = 200
        self.max_tokens_unified = 300
    
    async def summarize_document(self, content: str, filename: str) -> str:
        """Generate a summary for a single document"""
        prompt = f"""Summarize this document in 2-3 sentences, focusing on key decisions, actions, or findings:

Document: {filename}
Content: {content[:2000]}  # Limit content to avoid token limits

Summary:"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.max_tokens_individual,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            print(f"OpenAI API error for {filename}: {str(e)}")
            return f"Error generating summary for {filename}. Please try again."
    
    async def create_unified_summary(self, individual_summaries: List[Dict]) -> str:
        """Create a unified summary from multiple individual summaries"""
        summaries_text = "\n\n".join([
            f"From {item['filename']}: {item['summary']}" 
            for item in individual_summaries
        ])
        
        prompt = f"""Create a unified summary from these individual document summaries. Organize by themes and key points, and highlight any overlaps or potential contradictions you notice:

Individual Summaries:
{summaries_text}

Create a cohesive unified summary that:
1. Groups similar themes together
2. Notes any contradictions or inconsistencies
3. Highlights the most important decisions or actions across all documents

Unified Summary:"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.max_tokens_unified,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            print(f"OpenAI API error for unified summary: {str(e)}")
            return "Error generating unified summary. Please try again."