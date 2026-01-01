"""
Test State Immutability
========================

Verify that __slots__ prevents dynamic attributes and AnswerState
doesn't store generator in self.
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from finitestatemachineAgent.hfsm_agent_async import AsyncHierarchicalState, AnswerState
from unittest.mock import Mock


def test_state_cannot_add_attributes():
    """Verify __slots__ prevents dynamic attributes"""
    class TestState(AsyncHierarchicalState):
        __slots__ = ("parent",)
        
        async def handle(self, context):
            return None
    
    state = TestState()
    
    with pytest.raises(AttributeError):
        state.foo = "bar"  # Should fail due to __slots__


def test_answer_state_no_generator_attribute():
    """Verify AnswerState doesn't store generator in self"""
    llm_mock = Mock()
    answer_state = AnswerState(parent=None, llm=llm_mock)
    
    # Should not have generator attribute after init
    assert not hasattr(answer_state, 'generator')


def test_base_state_only_has_parent():
    """Verify base state only allows parent attribute"""
    class MinimalState(AsyncHierarchicalState):
        __slots__ = ("parent",)
        
        async def handle(self, context):
            return None
    
    state = MinimalState()
    
    # Should be able to set parent
    parent_state = MinimalState()
    state.parent = parent_state
    assert state.parent == parent_state
    
    # Should not be able to add other attributes
    with pytest.raises(AttributeError):
        state.some_data = "value"
