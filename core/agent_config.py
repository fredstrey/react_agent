"""
Agent Configuration
===================

Centralized configuration for AgentEngine with all tunable parameters.
"""

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class AgentConfig:
    """
    Configuration for AgentEngine.
    
    All parameters are tunable to customize agent behavior without code changes.
    """
    
    # ===== Retry Configuration =====
    max_retries: int = 3
    """Maximum number of retry attempts for failed tool calls"""
    
    retry_delay: float = 1.0
    """Delay in seconds between retry attempts"""
    
    # ===== Context Pruning =====
    pruning_strategy: str = "cut_last_n"
    """Strategy for pruning context: 'cut_last_n', 'cut_first_n', 'smart'"""
    
    max_context_tokens: int = 4000
    """Maximum tokens to keep in context before pruning"""
    
    # ===== Concurrency =====
    max_workers: int = 4
    """Maximum number of concurrent tool executions"""
    
    tool_timeout: float = 30.0
    """Timeout in seconds for individual tool execution"""
    
    # ===== State Timeouts =====
    state_timeouts: Dict[str, float] = field(default_factory=lambda: {
        "RouterState": 30.0,
        "ToolState": 60.0,
        "ValidationState": 20.0,
        "AnswerState": 120.0
    })
    """Timeout in seconds for each state type"""
    
    # ===== Circuit Breaker =====
    circuit_breaker_enabled: bool = True
    """Enable circuit breaker for tool execution"""
    
    circuit_breaker_threshold: int = 5
    """Number of failures before opening circuit"""
    
    circuit_breaker_timeout: float = 60.0
    """Time in seconds before attempting to close circuit"""
    
    # ===== Validation =====
    validation_prompt: str = (
        "Analyze the tool results and determine if they adequately answer "
        "the user's question. Respond with 'true' if sufficient, 'false' otherwise."
    )
    """Custom prompt for ValidationState"""

    skip_validation: bool = True
    """Skip validation step and proceed directly to AnswerState"""
    
    # ===== Logging & Observability =====
    enable_snapshots: bool = True
    """Save state snapshots for debugging"""
    
    snapshot_dir: str = "logs/snapshots"
    """Directory for saving snapshots"""
    
    log_level: str = "INFO"
    """Logging level: DEBUG, INFO, WARNING, ERROR"""
    
    # ===== Performance =====
    enable_caching: bool = False
    """Enable caching of tool results (experimental)"""
    
    cache_ttl: int = 300
    """Cache time-to-live in seconds"""
    
    def validate(self) -> None:
        """Validate configuration values."""
        if self.max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        
        if self.max_workers < 1:
            raise ValueError("max_workers must be >= 1")
        
        if self.pruning_strategy not in ["cut_last_n", "cut_first_n", "smart"]:
            raise ValueError(f"Invalid pruning_strategy: {self.pruning_strategy}")
        
        if self.tool_timeout <= 0:
            raise ValueError("tool_timeout must be > 0")
        
        for state, timeout in self.state_timeouts.items():
            if timeout <= 0:
                raise ValueError(f"Timeout for {state} must be > 0")


# Default configurations for common use cases
DEFAULT_CONFIG = AgentConfig()

FAST_CONFIG = AgentConfig(
    max_workers=8,
    tool_timeout=10.0,
    state_timeouts={
        "RouterState": 10.0,
        "ToolState": 20.0,
        "ValidationState": 5.0,
        "AnswerState": 30.0
    }
)

ROBUST_CONFIG = AgentConfig(
    max_retries=5,
    circuit_breaker_threshold=10,
    tool_timeout=60.0,
    enable_snapshots=True
)

MINIMAL_CONFIG = AgentConfig(
    max_workers=1,
    max_retries=1,
    circuit_breaker_enabled=False,
    enable_snapshots=False
)
