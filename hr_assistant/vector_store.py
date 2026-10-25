"""Step 4: store chunk embeddings in Qdrant Cloud so we can search them later."""
import os
from langchain_community.vectorstores import FAISS
from hr_assistant import config
from hr_assistant.embeddings import get_embeddings_model

#build vector store
def build_vector_store(chunks):
    """Embed every chunk and upload it into a FAISS vector store."""

    embeddings_model = get_embeddings_model()
    return FAISS.from_documents(chunks, embeddings_model)

#save vector store
def save_vector_store(vector_store, path: str = config.VECTOR_STORE_PATH)->None:
    """Save the vector store to disk."""
    vector_store.save_local(path)

#load vector store
def load_vector_store(path: str = config.VECTOR_STORE_PATH):
    """Load the vector store from disk."""
    embeddings_model = get_embeddings_model()
    return FAISS.load_local(path, embeddings_model,allow_dangerous_deserialization=True)

def vector_store_exists(path: str = config.VECTOR_STORE_PATH)->bool:
    """Check if the vector store exists on disk."""
    return os.path.exists(os.path.join(path, "index.faiss"))

def get_retriever(vector_store, k: int = config.TOP_K_RESULTS):
    """Return a retriever for the vector store."""
    return vector_store.as_retriever(search_kwargs={"k": k})