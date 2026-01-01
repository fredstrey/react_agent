"""
Test Token Accumulation in Forks
==================================

Verifies that token usage from forks is correctly accumulated back to the parent context.
"""

import sys
import os
# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock
from core.context_async import AsyncExecutionContext, SafetyMonitor
from finitestatemachineAgent.hfsm_agent_async import AsyncAgentEngine


async def test_token_accumulation():
    """Test that fork token usage is accumulated back to parent."""
    print("\n🧪 Test: Token Accumulation from Forks")
    print("=" * 60)
    
    # Mock LLM with token usage
    mock_llm = MagicMock()
    
    # Mock response for ResearchForkState (with token usage)
    mock_llm.chat_with_tools = AsyncMock(return_value={
        "tool_calls": [{
            "function": {
                "name": "search_web",
                "arguments": json.dumps({"query": "test query"})
            }
        }],
        "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
    })
    
    # Mock registry
    mock_registry = MagicMock()
    mock_registry.to_openai_format.return_value = [
        {
            "type": "function",
            "function": {
                "name": "search_web",
                "description": "Search the web",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"}
                    }
                }
            }
        }
    ]
    
    # Mock executor
    mock_executor = MagicMock()
    mock_executor.execute_parallel = AsyncMock(return_value=["Mock result"])
    
    # Create engine
    engine = AsyncAgentEngine(
        llm=mock_llm,
        registry=mock_registry,
        executor=mock_executor,
        enable_parallel_planning=True,
        skip_validation=True
    )
    
    # Create parent context
    monitor = SafetyMonitor(max_requests=50)
    parent_ctx = AsyncExecutionContext(user_query="Test query", safety_monitor=monitor)
    
    # Add some initial usage to parent
    await parent_ctx.accumulate_usage({"prompt_tokens": 50, "completion_tokens": 25, "total_tokens": 75})
    
    print(f"📊 Initial parent usage: {await parent_ctx.get_total_usage()}")
    
    # Create 2 forks
    fork1 = parent_ctx.fork()
    await fork1.set_memory("branch_id", "fork_1")
    await fork1.set_memory("branch_goal", "Research X")
    
    fork2 = parent_ctx.fork()
    await fork2.set_memory("branch_id", "fork_2")
    await fork2.set_memory("branch_goal", "Research Y")
    
    # Execute forks (simulate)
    for fork in [fork1, fork2]:
        # Execute ResearchForkState
        research_state = engine.research_fork_state
        await research_state.handle(fork)
        
        # Execute ToolState
        tool_state = engine.tool_state
        await tool_state.handle(fork)
        
        # Execute ForkSummaryState
        summary_state = engine.fork_summary_state
        await summary_state.handle(fork)
    
    print(f"📊 Fork 1 usage: {await fork1.get_total_usage()}")
    print(f"📊 Fork 2 usage: {await fork2.get_total_usage()}")
    
    # Simulate ForkDispatchState accumulation
    successful_forks = [fork1, fork2]
    total_fork_tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for fork_ctx in successful_forks:
        fork_usage = await fork_ctx.get_total_usage()
        total_fork_tokens["prompt_tokens"] += fork_usage.get("prompt_tokens", 0)
        total_fork_tokens["completion_tokens"] += fork_usage.get("completion_tokens", 0)
        total_fork_tokens["total_tokens"] += fork_usage.get("total_tokens", 0)
    
    print(f"📊 Total fork tokens: {total_fork_tokens}")
    
    # Accumulate to parent
    await parent_ctx.accumulate_usage(total_fork_tokens)
    
    final_usage = await parent_ctx.get_total_usage()
    print(f"📊 Final parent usage: {final_usage}")
    
    # Verify
    expected_total = 75 + (150 * 2)  # Initial + (fork1 + fork2)
    assert final_usage["total_tokens"] == expected_total, \
        f"❌ Expected {expected_total} total tokens, got {final_usage['total_tokens']}"
    
    print(f"✅ Token accumulation verified!")
    print(f"   - Initial: 75 tokens")
    print(f"   - Fork 1: 150 tokens")
    print(f"   - Fork 2: 150 tokens")
    print(f"   - Final: {final_usage['total_tokens']} tokens")


async def main():
    """Run test."""
    print("\n" + "=" * 60)
    print("🚀 Token Accumulation Test")
    print("=" * 60)
    
    try:
        await test_token_accumulation()
        
        print("\n" + "=" * 60)
        print("✅ Test passed!")
        print("=" * 60)
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        raise
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
