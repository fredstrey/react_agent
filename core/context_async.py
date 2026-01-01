"""
Async Execution Context
========================

Thread-safe execution context using asyncio.Lock instead of threading.RLock.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
import asyncio
import copy
import uuid

class SafetyLimitExceeded(Exception):
    """Raised when the LLM request limit is exceeded for a workflow."""
    pass

class SafetyMonitor:
    """Tracks LLM usage across a workflow execution (shared by forks)."""
    def __init__(self, max_requests: int = 50):
        self.count = 0
        self.max_requests = max_requests
        self._lock = asyncio.Lock()
        
    async def increment(self):
        async with self._lock:
            self.count += 1
            if self.count > self.max_requests:
                raise SafetyLimitExceeded(f"⛔ Safety Limit Exceeded: {self.max_requests} LLM requests reached.")


class AsyncExecutionContext(BaseModel):
    """
    Async execution context with asyncio.Lock for concurrency safety.
    
    All mutation methods are async and use lock for thread safety.
    """
    # Original user query
    user_query: str = Field(..., description="Original user query")
    
    # Parent context for forks (enables sub-contexts)
    parent: Optional['AsyncExecutionContext'] = Field(default=None, description="Parent context for forks")
    
    class Config:
        arbitrary_types_allowed = True
    
    def __init__(self, safety_monitor: Optional[SafetyMonitor] = None, **data):
        super().__init__(**data)
        # Async lock (not serialized)
        object.__setattr__(self, '_lock', asyncio.Lock())
        
        # Shared safety monitor logic
        if safety_monitor:
            object.__setattr__(self, 'safety_monitor', safety_monitor)
        elif self.parent and hasattr(self.parent, 'safety_monitor'):
            object.__setattr__(self, 'safety_monitor', self.parent.safety_monitor)
        else:
            # Default monitor if none provided (e.g. legacy init)
            object.__setattr__(self, 'safety_monitor', SafetyMonitor(max_requests=50))
            
    async def increment_llm_call(self):
        """Register an LLM call and check limit."""
        if hasattr(self, 'safety_monitor'):
            await self.safety_monitor.increment()
    
    # Memory for arbitrary data storage
    memory: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary memory")
    
    # Tool calls history
    tool_calls: List[Dict[str, Any]] = Field(default_factory=list, description="Tool calls history")
    
    # Execution state
    current_iteration: int = Field(default=0, description="Current iteration")
    max_iterations: int = Field(default=3, description="Maximum iterations")
    
    # Timestamp & Metrics
    timestamp: datetime = Field(default_factory=datetime.now, description="Execution timestamp")
    start_time: datetime = Field(default_factory=datetime.now, description="Start time")
    metrics: Dict[str, Any] = Field(default_factory=dict, description="Performance metrics")

    async def add_tool_call(self, tool_name: str, arguments: Dict[str, Any], result: Any = None):
        """Add a tool call to the history (async-safe)"""
        async with self._lock:
            self.tool_calls.append({
                "tool_name": tool_name,
                "arguments": arguments,
                "result": result,
                "iteration": self.current_iteration,
                "timestamp": datetime.now().isoformat()
            })

    async def set_memory(self, key: str, value: Any):
        """Store value in memory (async-safe)"""
        async with self._lock:
            self.memory[key] = value

    async def get_memory(self, key: str, default: Any = None) -> Any:
        """Retrieve value from memory (async-safe)"""
        async with self._lock:
            return self.memory.get(key, default)

    async def update_tool_results(self, pending: List[Dict], results: List[Any]):
        """Update tool results atomically (replaces direct _lock access)"""
        async with self._lock:
            for call, result in zip(pending, results):
                call["result"] = result

    async def increment_iteration(self) -> int:
        """Increment retry counter atomically"""
        async with self._lock:
            self.current_iteration += 1
            return self.current_iteration

    async def get_total_usage(self) -> Dict[str, int]:
        """Get accumulated token usage"""
        async with self._lock:
            return self.memory.get("total_usage", {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            })

    async def accumulate_usage(self, usage: Dict[str, int]):
        """Accumulate token usage atomically"""
        async with self._lock:
            total = self.memory.get("total_usage", {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            })
            total["prompt_tokens"] += usage.get("prompt_tokens", 0)
            total["completion_tokens"] += usage.get("completion_tokens", 0)
            total["total_tokens"] += usage.get("total_tokens", 0)
            self.memory["total_usage"] = total

    def fork(self) -> 'AsyncExecutionContext':
        """
        Create a child context with copied memory.
        Useful for speculative execution or parallel branches.
        """
        from copy import deepcopy
        
        child = AsyncExecutionContext(
            user_query=self.user_query,
            parent=self,
            safety_monitor=self.safety_monitor # Propagate shared monitor
        )
        
        # Deep copy memory to isolate mutations
        child.memory = deepcopy(self.memory)
        child.tool_calls = deepcopy(self.tool_calls)
        child.current_iteration = self.current_iteration
        child.max_iterations = self.max_iterations
        
        return child

    async def merge_from_child(self, child: 'AsyncExecutionContext'):
        """
        Merge successful child context results back to parent.
        """
        async with self._lock:
            # Merge tool calls
            self.tool_calls.extend(child.tool_calls)
            
            # Merge memory (child wins on conflicts)
            self.memory.update(child.memory)
            
            # Accumulate usage
            child_usage = child.memory.get("total_usage")
            if child_usage:
                await self.accumulate_usage(child_usage)

    def snapshot(self) -> Dict[str, Any]:
        """
        Create a serializable snapshot (sync method for compatibility).
        """
        return {
            "user_query": self.user_query,
            "memory": {k: str(v) for k, v in self.memory.items()},
            "tool_calls": self.tool_calls,
            "current_iteration": self.current_iteration,
            "max_iterations": self.max_iterations,
            "timestamp": self.timestamp.isoformat(),
            "start_time": self.start_time.isoformat(),
            "metrics": self.metrics,
            # 🔥 NEW: Include request count in snapshot
            "request_count": self.safety_monitor.count if hasattr(self, 'safety_monitor') else 0
        }
