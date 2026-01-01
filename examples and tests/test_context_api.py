"""
Test Context API Isolation
===========================

Verify that context methods properly encapsulate lock usage
and provide atomic operations.
"""

import sys
import os
import pytest
import asyncio

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.context_async import AsyncExecutionContext


@pytest.mark.asyncio
async def test_update_tool_results():
    """Test atomic tool result updates"""
    context = AsyncExecutionContext(user_query="test")
    
    pending = [
        {"tool_name": "tool1", "arguments": {}, "result": None},
        {"tool_name": "tool2", "arguments": {}, "result": None}
    ]
    
    results = ["result1", "result2"]
    
    await context.update_tool_results(pending, results)
    
    assert pending[0]["result"] == "result1"
    assert pending[1]["result"] == "result2"


@pytest.mark.asyncio
async def test_increment_iteration():
    """Test atomic iteration increment"""
    context = AsyncExecutionContext(user_query="test")
    
    assert context.current_iteration == 0
    
    new_iter = await context.increment_iteration()
    assert new_iter == 1
    assert context.current_iteration == 1
    
    # Test multiple increments
    await context.increment_iteration()
    await context.increment_iteration()
    assert context.current_iteration == 3


@pytest.mark.asyncio
async def test_accumulate_usage():
    """Test atomic token usage accumulation"""
    context = AsyncExecutionContext(user_query="test")
    
    await context.accumulate_usage({
        "prompt_tokens": 10,
        "completion_tokens": 20,
        "total_tokens": 30
    })
    
    await context.accumulate_usage({
        "prompt_tokens": 5,
        "completion_tokens": 10,
        "total_tokens": 15
    })
    
    total = await context.get_total_usage()
    
    assert total["prompt_tokens"] == 15
    assert total["completion_tokens"] == 30
    assert total["total_tokens"] == 45


@pytest.mark.asyncio
async def test_get_total_usage_default():
    """Test get_total_usage returns default when not set"""
    context = AsyncExecutionContext(user_query="test")
    
    total = await context.get_total_usage()
    
    assert total["prompt_tokens"] == 0
    assert total["completion_tokens"] == 0
    assert total["total_tokens"] == 0


@pytest.mark.asyncio
async def test_concurrent_accumulation():
    """Test that concurrent usage accumulation is safe"""
    context = AsyncExecutionContext(user_query="test")
    
    async def accumulate_batch():
        for _ in range(10):
            await context.accumulate_usage({
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "total_tokens": 2
            })
    
    # Run 5 concurrent batches
    await asyncio.gather(*[accumulate_batch() for _ in range(5)])
    
    total = await context.get_total_usage()
    
    # Should have accumulated 50 times
    assert total["prompt_tokens"] == 50
    assert total["completion_tokens"] == 50
    assert total["total_tokens"] == 100
