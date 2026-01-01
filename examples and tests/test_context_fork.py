"""
Test Context Fork Support
==========================

Verify that context forking creates isolated copies and
merge operations work correctly.
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.context_async import AsyncExecutionContext


@pytest.mark.asyncio
async def test_fork_creates_isolated_copy():
    """Test that fork creates isolated context"""
    parent = AsyncExecutionContext(user_query="test")
    await parent.set_memory("key1", "value1")
    
    child = parent.fork()
    
    # Child should have copied memory
    child_val = await child.get_memory("key1")
    assert child_val == "value1"
    
    # Mutations should be isolated
    await child.set_memory("key1", "modified")
    await child.set_memory("key2", "new")
    
    parent_val = await parent.get_memory("key1")
    assert parent_val == "value1"  # Parent unchanged
    
    parent_key2 = await parent.get_memory("key2")
    assert parent_key2 is None  # Parent doesn't have child's new key


@pytest.mark.asyncio
async def test_fork_copies_tool_calls():
    """Test that fork copies tool calls"""
    parent = AsyncExecutionContext(user_query="test")
    await parent.add_tool_call("tool1", {"arg": "value"}, "result1")
    
    child = parent.fork()
    
    # Child should have copied tool calls
    assert len(child.tool_calls) == 1
    assert child.tool_calls[0]["tool_name"] == "tool1"
    
    # Mutations should be isolated
    await child.add_tool_call("tool2", {"arg": "value2"}, "result2")
    
    assert len(child.tool_calls) == 2
    assert len(parent.tool_calls) == 1  # Parent unchanged


@pytest.mark.asyncio
async def test_fork_copies_iteration_state():
    """Test that fork copies iteration state"""
    parent = AsyncExecutionContext(user_query="test")
    await parent.increment_iteration()
    await parent.increment_iteration()
    
    child = parent.fork()
    
    assert child.current_iteration == 2
    assert child.max_iterations == parent.max_iterations
    
    # Child mutations don't affect parent
    await child.increment_iteration()
    assert child.current_iteration == 3
    assert parent.current_iteration == 2


@pytest.mark.asyncio
async def test_merge_from_child():
    """Test merging child context back to parent"""
    parent = AsyncExecutionContext(user_query="test")
    await parent.set_memory("key1", "original")
    
    child = parent.fork()
    await child.set_memory("key1", "modified")
    await child.set_memory("key2", "new")
    
    await parent.merge_from_child(child)
    
    assert await parent.get_memory("key1") == "modified"
    assert await parent.get_memory("key2") == "new"


@pytest.mark.asyncio
async def test_merge_tool_calls():
    """Test merging tool calls from child"""
    parent = AsyncExecutionContext(user_query="test")
    await parent.add_tool_call("tool1", {}, "result1")
    
    child = parent.fork()
    await child.add_tool_call("tool2", {}, "result2")
    
    await parent.merge_from_child(child)
    
    # Parent should have both tool calls
    assert len(parent.tool_calls) == 3  # original + child's copy + child's new
    tool_names = [call["tool_name"] for call in parent.tool_calls]
    assert "tool1" in tool_names
    assert "tool2" in tool_names


@pytest.mark.asyncio
async def test_merge_accumulates_usage():
    """Test that merge accumulates token usage"""
    parent = AsyncExecutionContext(user_query="test")
    await parent.accumulate_usage({
        "prompt_tokens": 10,
        "completion_tokens": 20,
        "total_tokens": 30
    })
    
    child = parent.fork()
    await child.accumulate_usage({
        "prompt_tokens": 5,
        "completion_tokens": 10,
        "total_tokens": 15
    })
    
    await parent.merge_from_child(child)
    
    total = await parent.get_total_usage()
    
    # Should have accumulated both parent and child usage
    assert total["prompt_tokens"] == 20  # 10 + 5 + 5 (child had parent's 10 in fork)
    assert total["completion_tokens"] == 40  # 20 + 10 + 10
    assert total["total_tokens"] == 60  # 30 + 15 + 15


@pytest.mark.asyncio
async def test_fork_preserves_parent_reference():
    """Test that forked context maintains parent reference"""
    parent = AsyncExecutionContext(user_query="test")
    child = parent.fork()
    
    assert child.parent == parent
    assert parent.parent is None
