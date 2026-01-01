"""
Explicit Transition Semantics
==============================

Represents state transitions with metadata for logging and tracing.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class Transition:
    """
    Explicit state transition with metadata.
    
    Attributes:
        to: Target state type name (e.g., "ToolState")
        reason: Human-readable reason for transition
        metadata: Additional context for logging/tracing
    """
    to: str
    reason: str
    metadata: Optional[dict] = None
    
    def __repr__(self):
        return f"Transition(to={self.to}, reason={self.reason})"
