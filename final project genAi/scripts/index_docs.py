import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import PyPDFLoader
# Paths
DATA_DIR = os.path.join(os.path.dirname(__file__), "../data")
PERSIST_DIR = os.path.join(os.path.dirname(__file__), "../chroma_db")

# Load all PDFs
documents = []
for file_name in os.listdir(DATA_DIR):
    if file_name.endswith(".pdf"):
        loader = PyPDFLoader(os.path.join(DATA_DIR, file_name))
        documents.extend(loader.load())

# Create embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Store in Chroma
vectordb = Chroma.from_documents(documents, embeddings, persist_directory=PERSIST_DIR)
vectordb.persist()
print(f"Ingested {len(documents)} documents into {PERSIST_DIR}")
