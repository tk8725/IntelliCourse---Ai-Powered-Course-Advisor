# IntelliCourse---Ai-Powered-Course-Advisor

**What it is**  
IntelliCourse is an AI-powered project that implements a Retrieval-Augmented Generation (RAG) course advisor using FastAPI, LangChain, and LangGraph. It answers course-catalog questions by searching indexed PDFs, and falls back to web search for general questions.

## Features
- Index PDF course catalogs into a local Chroma DB
- LangGraph agent routes queries to catalog retriever or web search
- FastAPI `/chat` endpoint for integration
- Minimal frontend to demo queries
- Supports both course-specific and general questions

## Project Structure

intellicourse/
├─ app/
│ └─ main.py # FastAPI app and endpoints
├─ rag/
│ ├─ indexer.py # PDF -> chunks -> embeddings -> Chroma
│ └─ retriever.py # Load Chroma retriever
├─ agent/
│ └─ langgraph_agent.py # LangGraph graph: router -> retriever/web -> generate
├─ core/
│ └─ llm_provider.py # Wrap LLM selection (OpenAI or HF)
├─ scripts/
│ └─ index_docs.py # CLI script to index PDFs into Chroma
├─ frontend/
│ └─ index.html # Simple UI
├
└─ requirements.txt

## Setup (Local)

1. Clone the repo and navigate to the project folder:

git clone <your-repo-url>
cd intellicourse

2. Create and activate a virtual environment:

python -m venv .venv
# Windows
.venv\Scripts\activate

3. Install dependencies:

pip install -r requirements.txt

4. Add your PDF course catalogs to a folder, e.g., data/.

5. Index your PDFs into Chroma DB:

python -m scripts.index_docs

6. Start the FastAPI server:

uvicorn app.main:app --reload

7. Open the frontend:

http://127.0.0.1:8000/static/index.html

8. Test queries via /chat endpoint:

POST /chat
{
  "query": "What are the prerequisites of Machine Learning?"
}
