
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import logging

logger = logging.getLogger(__name__)

class RetrieverWrapper:
    """Wrapper around Chroma retriever for LangGraph compatibility."""
    def __init__(self, persist_dir="chroma_db"):
        try:
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            self.vectordb = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
            self.retriever = self.vectordb.as_retriever()
            logger.info("Retriever initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing retriever: {e}")
            raise

    def get_docs(self, query: str):
        """LangGraph calls this to get relevant docs."""
        try:
            docs = self.retriever.get_relevant_documents(query)
            logger.info(f"Retrieved {len(docs)} documents for query: {query}")
            return docs
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return []

# Initialize retriever
retriever = RetrieverWrapper()