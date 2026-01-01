"""
Test Transition Semantics
==========================

Verify that Transition objects work correctly for explicit
state transitions with metadata.
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from finitestatemachineAgent.transition import Transition


def test_transition_creation():
    """Test Transition object creation"""
    t = Transition(to="ToolState", reason="tool_calls", metadata={"count": 3})
    
    assert t.to == "ToolState"
    assert t.reason == "tool_calls"
    assert t.metadata["count"] == 3


def test_transition_without_metadata():
    """Test Transition creation without metadata"""
    t = Transition(to="AnswerState", reason="no_tools")
    
    assert t.to == "AnswerState"
    assert t.reason == "no_tools"
    assert t.metadata is None


def test_transition_repr():
    """Test Transition string representation"""
    t = Transition(to="AnswerState", reason="no_tools")
    
    repr_str = repr(t)
    assert "AnswerState" in repr_str
    assert "no_tools" in repr_str


def test_transition_with_complex_metadata():
    """Test Transition with complex metadata"""
    metadata = {
        "tool_count": 3,
        "tools": ["search", "calculator", "weather"],
        "confidence": 0.95
    }
    
    t = Transition(to="ToolState", reason="multiple_tools", metadata=metadata)
    
    assert t.metadata["tool_count"] == 3
    assert len(t.metadata["tools"]) == 3
    assert t.metadata["confidence"] == 0.95


def test_transition_equality():
    """Test that transitions with same values are equal"""
    t1 = Transition(to="ToolState", reason="test")
    t2 = Transition(to="ToolState", reason="test")
    
    # Dataclass equality
    assert t1 == t2


def test_transition_inequality():
    """Test that transitions with different values are not equal"""
    t1 = Transition(to="ToolState", reason="test1")
    t2 = Transition(to="ToolState", reason="test2")
    
    assert t1 != t2
