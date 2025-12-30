"""
Execution context for RAG Agent
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


class ExecutionContext(BaseModel):
    """
    Execution context that maintains state during processing
    """
    # Original user query (immutable)
    user_query: str = Field(..., description="Original user query")
    
    # Chat history
    chat_history: List[Dict[str, str]] = Field(default_factory=list, description="Chat history")
    
    # Execution metadata
    conversation_id: Optional[str] = Field(None, description="Conversation ID")
    timestamp: datetime = Field(default_factory=datetime.now, description="Execution timestamp")
    
    # Execution state
    current_iteration: int = Field(default=0, description="Current iteration")
    max_iterations: int = Field(default=3, description="Maximum iterations")
    
    # Retrieved documents
    retrieved_documents: List[Dict[str, Any]] = Field(default_factory=list, description="Retrieved documents")
    sources_used: List[str] = Field(default_factory=list, description="Sources used")
    
    # Tool calls history
    tool_calls_history: List[Dict[str, Any]] = Field(default_factory=list, description="Tool calls history")
    
    # State flags
    is_out_of_scope: bool = Field(default=False, description="If the question is out of scope")
    has_context: bool = Field(default=False, description="If context was retrieved")
    
    class Config:
        arbitrary_types_allowed = True
    
    def add_tool_call(self, tool_name: str, arguments: Dict[str, Any], result: Any = None):
        """Add a tool call to the history"""
        self.tool_calls_history.append({
            "tool_name": tool_name,
            "arguments": arguments,
            "result": result,
            "iteration": self.current_iteration
        })
    
    def add_document(self, content: str, score: float, metadata: Dict[str, Any]):
        """Add a retrieved document"""
        self.retrieved_documents.append({
            "content": content,
            "score": score,
            "metadata": metadata
        })
        self.has_context = True
        
        # Add source if available
        if "source" in metadata:
            source = metadata["source"]
            if source not in self.sources_used:
                self.sources_used.append(source)
    
    def mark_out_of_scope(self):
        """Mark the question as out of scope"""
        self.is_out_of_scope = True
    
    def get_context_summary(self) -> str:
        """Return a context summary"""
        return f"""
Contexto de Execução:
- Query: {self.user_query}
- Iteração: {self.current_iteration}/{self.max_iterations}
- Documents retrieved: {len(self.retrieved_documents)}
- Sources: {len(self.sources_used)}
- Tool calls: {len(self.tool_calls_history)}
- Out of scope: {self.is_out_of_scope}
"""
