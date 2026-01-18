
import os
from langchain_google_genai import ChatGoogleGenerativeAI

def get_llm(model_name: str):
    """Get LLM instance using free Gemini 1.5 Flash."""
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set. Get a free key from: https://aistudio.google.com/app/apikey")
    
    
    model_id = "gemini-2.5-flash"
    
    print(f" Initializing Gemini model: {model_id}")  
    
    try:
        llm = ChatGoogleGenerativeAI(
            model=model_id,
            google_api_key=google_api_key,
            temperature=0.1,
            max_tokens=1000
        )
        print(" Gemini 1.5 Flash initialized successfully")
        return llm
    except Exception as e:
        print(f" Error initializing {model_id}: {e}")
        
        # Fallback to Gemini 1.0 Pro
        try:
            print(" Trying fallback: gemini-1.0-pro")
            return ChatGoogleGenerativeAI(
                model="gemini-1.0-pro",
                google_api_key=google_api_key,
                temperature=0.1,
                max_tokens=1000
            )
        except Exception as e2:
            raise ValueError(f"Both Gemini models failed: {e2}")