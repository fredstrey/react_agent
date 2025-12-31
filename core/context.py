from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime

class ExecutionContext(BaseModel):
    """
    Generic execution context that maintains state during processing
    """
    # Original user query
    user_query: str = Field(..., description="Original user query")
    
    # Memory for arbitrary data storage (e.g., intermediate results, settings)
    memory: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary memory for execution")
    
    # Tool calls history
    tool_calls: List[Dict[str, Any]] = Field(default_factory=list, description="Tool calls history")
    
    # Execution state
    current_iteration: int = Field(default=0, description="Current iteration")
    max_iterations: int = Field(default=3, description="Maximum iterations")
    
    # Timestamp & Metrics
    timestamp: datetime = Field(default_factory=datetime.now, description="Execution timestamp")
    start_time: datetime = Field(default_factory=datetime.now, description="Start execution time")
    metrics: Dict[str, Any] = Field(default_factory=dict, description="Performance metrics")

    def add_tool_call(self, tool_name: str, arguments: Dict[str, Any], result: Any = None):
        """Add a tool call to the history"""
        self.tool_calls.append({
            "tool_name": tool_name,
            "arguments": arguments,
            "result": result,
            "iteration": self.current_iteration,
            "timestamp": datetime.now().isoformat()
        })

    def set_memory(self, key: str, value: Any):
        """Store value in memory"""
        self.memory[key] = value

    def get_memory(self, key: str, default: Any = None) -> Any:
        """Retrieve value from memory"""
        return self.memory.get(key, default)

    def snapshot(self) -> Dict[str, Any]:
        """
        Create a serializable snapshot of the current context.
        Excludes non-serializable objects from memory if possible, 
        or assumes memory is largely JSON-safe.
        """
        # We use model_dump(mode='json') to handle datetime and basic types
        data = self.model_dump(mode='json')
        
        # Manually filter known non-serializable keys from memory to be safe
        # (e.g. llm client, executor references if stored there)
        if "memory" in data:
            keys_to_remove = ["llm", "registry", "executor", "thread_pool"]
            for k in keys_to_remove:
                data["memory"].pop(k, None)
                
        return data
    @classmethod
    def load_from_snapshot(cls, snapshot: Dict[str, Any]) -> 'ExecutionContext':
        """
        Create an ExecutionContext instance from a snapshot dictionary.
        """
        # Pydantic automatically handles type coercion for datetime fields etc.
        return cls(**snapshot)
