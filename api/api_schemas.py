from pydantic import BaseModel, Field
from typing import List, Optional


class ChatMessage(BaseModel):
    """Chat message"""
    role: str = Field(..., description="Role: user or assistant")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    """Request for chat endpoint"""
    message: str = Field(..., description="User message")
    conversation_id: Optional[str] = Field(None, description="Conversation ID")
    stream: bool = Field(default=True, description="If streaming should be used")
    chat_history: Optional[List[ChatMessage]] = Field(default=None, description="Chat history (last 3 interactions)")


class ChatStreamChunk(BaseModel):
    """Chunk of streaming"""
    type: str = Field(..., description="Type: thinking, tool_call, response, done")
    content: str = Field(default="", description="Chunk content")
    metadata: Optional[dict] = Field(None, description="Additional metadata")


class ChatResponse(BaseModel):
    """Complete chat response"""
    answer: str = Field(..., description="Final answer")
    sources_used: List[str] = Field(default_factory=list, description="Sources used")
    confidence: Optional[str] = Field(None, description="Confidence level")
    conversation_id: str = Field(..., description="Conversation ID")


class ProcessPDFRequest(BaseModel):
    """Request to process PDF"""
    pdf_path: str = Field(..., description="Path to the PDF file")
    max_tokens: int = Field(default=500, description="Maximum tokens per chunk")


class ProcessPDFResponse(BaseModel):
    """Response from PDF processing"""
    status: str = Field(..., description="Processing status")
    pdf_file: str = Field(..., description="PDF file name")
    total_chunks: int = Field(..., description="Total of chunks created")
    total_tokens: int = Field(..., description="Total of tokens processed")
    avg_tokens_per_chunk: float = Field(..., description="Average tokens per chunk")
