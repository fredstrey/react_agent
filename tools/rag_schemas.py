from pydantic import BaseModel, Field
from typing import List, Optional


# =========================
# SCHEMAS FOR TOOLS
# =========================

class SearchArgs(BaseModel):
    """Arguments for semantic search in documents"""
    query: str = Field(..., description="Text of the query to search relevant documents")
    top_k: int = Field(default=3, description="Number of most relevant documents to return")


class DocumentChunk(BaseModel):
    """Document chunk returned by semantic search"""
    content: str = Field(..., description="Content of the document chunk")
    score: float = Field(..., description="Similarity score (0-1)")
    metadata: dict = Field(default_factory=dict, description="Document metadata")


class SearchResponse(BaseModel):
    """Response from semantic search tool"""
    query: str = Field(..., description="Original query")
    results: List[DocumentChunk] = Field(..., description="List of found documents")
    total_found: int = Field(..., description="Total number of documents found")


# =========================
# SCHEMA FOR FINAL RESPONSE
# =========================

class RAGResponse(BaseModel):
    """Final response validated by RAG agent"""
    answer: str = Field(..., description="User's question answer")
    sources_used: List[str] = Field(
        default_factory=list,
        description="List of sources/documents used to generate the response"
    )
    confidence: Optional[str] = Field(
        None,
        description="Confidence level in the response (high/medium/low)"
    )


# =========================
# SCHEMAS FOR PDF PROCESSING
# =========================

class ProcessPDFRequest(BaseModel):
    """Request to process PDF"""
    pdf_path: str = Field(..., description="Path to the PDF file")
    max_tokens: int = Field(default=500, description="Maximum tokens per chunk")
    overlap_tokens: int = Field(default=50, description="Tokens overlap between chunks")


class ProcessPDFResponse(BaseModel):
    """Response from PDF processing"""
    status: str = Field(..., description="Status of the processing")
    pdf_file: str = Field(..., description="Name of the PDF file")
    total_chunks: int = Field(..., description="Total number of chunks created")
    total_tokens: int = Field(..., description="Total number of tokens processed")
    avg_tokens_per_chunk: float = Field(..., description="Average tokens per chunk")
