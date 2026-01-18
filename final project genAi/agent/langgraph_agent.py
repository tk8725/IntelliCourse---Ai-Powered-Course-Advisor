
from langgraph.graph import StateGraph, END
from typing import TypedDict, Literal
from rag.retriever import retriever
from core.llm_provider import get_llm
from langchain_community.tools.tavily_search import TavilySearchResults
import os
from dotenv import load_dotenv

load_dotenv()

# -------- Agent State --------
class AgentState(TypedDict):
    query: str
    context: str
    answer: str
    source_tool: str
    next_node: str  # 

# -------- Components --------
try:
    llm = get_llm("gemini")
    tavily_tool = TavilySearchResults(api_key=os.getenv("TAVILY_API_KEY"), max_results=3)
except Exception as e:
    print(f"Error initializing components: {e}")
    llm = None
    tavily_tool = None

# -------- Router --------
def router_node(state: AgentState) -> AgentState:
    """Router node determines the next step and returns updated state."""
    q = state["query"].lower()
    
    if "course" in q or "prerequisite" in q or "catalog" in q:
        next_node = "course"
    else:
        next_node = "web"
    
    print(f"router node: {next_node}")
    
    #  Return updated state with next_node information
    return {
        **state,
        "next_node": next_node
    }

# -------- Conditional Edge Function --------
def route_condition(state: AgentState) -> Literal["course", "web"]:
    """This function returns the next node name based on state."""
    return state["next_node"]

# -------- Course Retriever --------
def course_node(state: AgentState) -> AgentState:
    """Retrieve course information."""
    print("course node: checking retriever")
    try:
        docs = retriever.get_docs(state["query"])
        if docs:
            context = "\n".join([d.page_content for d in docs])
        else:
            context = "No relevant course information found."
        
        return {
            **state,
            "context": context,
            "source_tool": "CourseCatalog"
        }
    except Exception as e:
        print(f"Error in course_node: {e}")
        return {
            **state,
            "context": f"Error retrieving course information: {str(e)}",
            "source_tool": "CourseCatalog_Error"
        }

# -------- Web Search --------
def web_node(state: AgentState) -> AgentState:
    """Search web for information."""
    try:
        if tavily_tool is None:
            raise ValueError("Tavily tool not initialized")
            
        results = tavily_tool.run(state["query"])
        if isinstance(results, list):
            context = "\n".join([r.get("content", "") for r in results])
        else:
            context = str(results)
        
        return {
            **state,
            "context": context,
            "source_tool": "TavilySearch"
        }
    except Exception as e:
        print(f"Error in web_node: {e}")
        return {
            **state,
            "context": f"Error searching web: {str(e)}",
            "source_tool": "TavilySearch_Error"
        }

# -------- Answer Generator --------
def generate_node(state: AgentState) -> AgentState:
    """Generate final answer."""
    try:
        if llm is None:
            raise ValueError("LLM not initialized")
            
        prompt = f"""
        You are IntelliCourse, an assistant for answering questions.
        
        Context:
        {state.get('context', '')}

        Question: {state.get('query', '')}

        Answer clearly and concisely based on the context provided.
        If the context doesn't contain relevant information, say so politely.
        """
        
        resp = llm.invoke(prompt)
        
        # Safely get content
        if hasattr(resp, "content"):
            answer = resp.content
        elif isinstance(resp, str):
            answer = resp
        else:
            answer = str(resp)
            
        return {
            **state,
            "answer": answer,
            "source_tool": state.get("source_tool", "unknown")
        }
        
    except Exception as e:
        print(f"Error in generate_node: {e}")
        return {
            **state,
            "answer": f"I encountered an error while generating a response: {str(e)}",
            "source_tool": "LLM_Error"
        }

# -------- Build Graph --------
def build_agent():
    """Build and return the LangGraph agent."""
    try:
        graph = StateGraph(AgentState)

        # Add nodes
        graph.add_node("router", router_node)  
        graph.add_node("course", course_node)
        graph.add_node("web", web_node)
        graph.add_node("generate", generate_node)

        # Set up the graph flow
        graph.set_entry_point("router")
        
        
        graph.add_conditional_edges(
            "router",
            route_condition,  
            {
                "course": "course",
                "web": "web"
            }
        )
        
        # Connect nodes
        graph.add_edge("course", "generate")
        graph.add_edge("web", "generate")
        graph.add_edge("generate", END)

        compiled_graph = graph.compile()
        print("LangGraph agent compiled successfully")
        return compiled_graph
        
    except Exception as e:
        print(f"Error building agent graph: {e}")
        raise