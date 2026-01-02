"""
RAG Tools using the framework pattern with decorators
"""
from typing import Dict, Any
from pydantic import BaseModel, Field
import logging

# Configure logger
logger = logging.getLogger("RAGTools")

from core.decorators import tool
from embedding_manager.embedding_manager import EmbeddingManager


# =========================
# SCHEMAS
# =========================

class SearchArgs(BaseModel):
    """Search arguments"""
    query: str = Field(..., description="Query de busca")
    top_k: int = Field(default=3, description="Number of results")



# =========================
# TOOLS
# =========================

# Variável global para armazenar o embedding_manager
_embedding_manager = None


def initialize_rag_tools(embedding_manager: EmbeddingManager):
    """
    Initialize Legal AI tools with the embedding manager
    
    Args:
        embedding_manager: EmbeddingManager instance
    """
    global _embedding_manager
    _embedding_manager = embedding_manager


@tool(
    name="search_documents",
    description="Search relevant documents in the knowledge base using semantic search"
)
def search_documents(query: str) -> Dict[str, Any]:
    """
    Search relevant documents in the knowledge base
    
    Args:
        query: User query
        top_k: Number of results to return
        
    Returns:
        Dictionary with search results
    """
    if _embedding_manager is None:
        return {
            "success": False,
            "error": "EmbeddingManager not initialized. Call initialize_rag_tools() first."
        }
    
    
    
    try:
        # DEBUG: Log da query recebida
        logger.info(f"🔍 [Search] search_documents called with query: '{query}'")

        # Search in Qdrant
        results = _embedding_manager.search(query=query, top_k=5)
        
        
        logger.info(f"✅ [Search] EmbeddingManager returned: {len(results)} results")
        
        # Format response
        chunks = [
            {
                "content": r["content"],
                "score": r["score"],
                "metadata": r["metadata"]
            }
            for r in results
        ]
        
        # Debug: Print chunks
        if chunks:
            logger.debug(f"📄 [Search] {len(chunks)} chunks found")
            for i, chunk in enumerate(chunks, 1):
                logger.debug(f"   Chunk {i} Score: {chunk['score']:.4f}")
        
        return {
            "success": True,
            "query": query,
            "results": chunks,
            "total_found": len(chunks)
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "query": query
        }