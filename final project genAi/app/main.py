
import os
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List
from agent.langgraph_agent import build_agent

# -------- Initialize FastAPI --------
app = FastAPI(title="IntelliCourse - Course Advisor API")

# Build LangGraph agent on startup
try:
    GRAPH = build_agent()
    print("LangGraph agent built successfully")
except Exception as e:
    print(f"Error building agent: {e}")
    GRAPH = None

# -------- Data Models --------
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    source_tool: str
    retrieved_context: List[str] = []

# -------- Frontend Configuration --------
frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")

# Check if frontend directory exists
if os.path.exists(frontend_dir) and os.path.isdir(frontend_dir):
    print(" Frontend directory found")
    
    # Mount static files (CSS, JS, images)
    static_dir = os.path.join(frontend_dir, "static")
    if os.path.exists(static_dir):
        app.mount("/static", StaticFiles(directory=static_dir), name="static")
        print(" Static files mounted")
    else:
        print(" Static directory not found, mounting frontend directly")
        app.mount("/static", StaticFiles(directory=frontend_dir), name="static")
    
    # Serving the main HTML file 
    @app.get("/")
    async def serve_frontend():
        index_path = os.path.join(frontend_dir, "index.html")
        if os.path.exists(index_path):
            return FileResponse(index_path)
        else:
          
            return {
                "message": "Frontend index.html not found. Server is running!",
                "api_endpoints": {
                    "chat": "/chat",
                    "health": "/health"
                }
            }
    
else:
    print(" Frontend directory not found")
    
    # Fallback root route when no frontend
    @app.get("/")
    def root():
        return {
            "message": "Server is running! Frontend not found.",
            "api_endpoints": {
                "chat": "/chat (POST with {'query': 'your question'})",
                "health": "/health"
            }
        }

# -------- API Health Check --------
@app.get("/health")
def health_check():
    """Health check endpoint to verify agent is ready"""
    if GRAPH is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    return {"status": "healthy", "agent_initialized": GRAPH is not None}

# -------- Chat Endpoint --------
@app.post("/chat", response_model=QueryResponse)
def chat(req: QueryRequest):
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="Query is empty")
    
    if GRAPH is None:
        raise HTTPException(status_code=503, detail="Agent not initialized. Please check server logs.")

    print("Received request:", req.dict())

    # Prepare initial state
    state = {
        "query": req.query.strip(),
        "context": "",
        "answer": "",
        "source_tool": "",
        "next_node": ""
    }

    try:
        result = GRAPH.invoke(state)
        print("GRAPH returned state:", result)
        
        answer = result.get("answer", "").strip()
        source_tool = result.get("source_tool", "unknown")
        context = result.get("context", "")
        
        if not answer:
            answer = "I couldn't generate a response. Please try rephrasing your question."
            
        retrieved_context = []
        if context:
            retrieved_context = [line.strip() for line in context.split("\n") if line.strip()]
        
        return QueryResponse(
            answer=answer,
            source_tool=source_tool,
            retrieved_context=retrieved_context
        )
        
    except Exception as e:
        print("GRAPH.invoke() error:", str(e))
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

# Error handling for common issues
@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    return HTTPException(status_code=400, detail=str(exc))

@app.exception_handler(TypeError)
async def type_error_handler(request, exc):
    return HTTPException(status_code=400, detail=str(exc))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")