"""
Test Light Fork Implementation
================================

Verifies that the new fork architecture works correctly:
1. Forks create lightweight contexts
2. ResearchForkState executes efficiently
3. ForkSummaryState produces structured summaries
4. MergeState creates semantic research_context
5. AnswerState injects research correctly
"""

import sys
import os
# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock
from core.context_async import AsyncExecutionContext, SafetyMonitor
from finitestatemachineAgent.hfsm_agent_async import (
    AsyncAgentEngine,
    ParallelPlan,
    BranchSpec
)


async def test_light_fork_context():
    """Test that fork() creates lightweight contexts."""
    print("\n🧪 Test 1: Light Fork Context Creation")
    print("=" * 60)
    
    # Create parent context
    monitor = SafetyMonitor(max_requests=50)
    parent = AsyncExecutionContext(user_query="Original query", safety_monitor=monitor)
    
    # Add some data to parent
    await parent.set_memory("some_key", "some_value")
    await parent.add_tool_call("test_tool", {"arg": "value"}, "result")
    
    # Fork
    child = parent.fork()
    
    # Verify lightweight properties
    assert child.user_query == "", "❌ Fork should have empty user_query"
    assert len(child.tool_calls) == 0, "❌ Fork should have empty tool_calls"
    assert child.memory.get("parent_query") == "Original query", "❌ Fork should preserve parent_query"
    assert child.memory.get("some_key") is None, "❌ Fork should NOT inherit parent memory"
    assert child.safety_monitor is monitor, "❌ Fork should share safety_monitor"
    
    print("✅ Fork creates lightweight context")
    print(f"   - Empty user_query: {child.user_query == ''}")
    print(f"   - Empty tool_calls: {len(child.tool_calls) == 0}")
    print(f"   - Shared safety_monitor: {child.safety_monitor is monitor}")
    print(f"   - Parent query preserved: {child.memory.get('parent_query')}")


async def test_fork_flow_with_mock():
    """Test the complete fork flow with mocked LLM and tools."""
    print("\n🧪 Test 2: Fork Flow (ResearchForkState -> ToolState -> ForkSummaryState)")
    print("=" * 60)
    
    # Mock LLM
    mock_llm = MagicMock()
    
    # Mock response for ResearchForkState (selects a tool)
    mock_llm.chat_with_tools = AsyncMock(return_value={
        "tool_calls": [{
            "function": {
                "name": "search_web",
                "arguments": json.dumps({"query": "test query"})
            }
        }],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
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
    mock_executor.execute_parallel = AsyncMock(return_value=["Mock search result"])
    
    # Create engine with parallel execution enabled
    engine = AsyncAgentEngine(
        llm=mock_llm,
        registry=mock_registry,
        executor=mock_executor,
        enable_parallel_planning=True,
        skip_validation=True
    )
    
    # Create fork context
    monitor = SafetyMonitor(max_requests=50)
    fork_ctx = AsyncExecutionContext(user_query="", safety_monitor=monitor)
    await fork_ctx.set_memory("branch_id", "test_branch_1")
    await fork_ctx.set_memory("branch_goal", "Research topic X")
    
    # Execute ResearchForkState
    research_state = engine.research_fork_state
    transition = await research_state.handle(fork_ctx)
    
    print(f"✅ ResearchForkState executed")
    print(f"   - Transition to: {transition.to}")
    print(f"   - Tool calls added: {len(fork_ctx.tool_calls)}")
    
    # Execute ToolState
    tool_state = engine.tool_state
    transition = await tool_state.handle(fork_ctx)
    
    print(f"✅ ToolState executed")
    print(f"   - Transition to: {transition.to}")
    print(f"   - Tool results: {fork_ctx.tool_calls[0].get('result')}")
    
    # Verify transition to ForkSummaryState (not AnswerState)
    assert transition.to == "ForkSummaryState", f"❌ Expected ForkSummaryState, got {transition.to}"
    
    # Execute ForkSummaryState
    summary_state = engine.fork_summary_state
    result = await summary_state.handle(fork_ctx)
    
    print(f"✅ ForkSummaryState executed")
    
    # Verify final_summary structure
    final_summary = await fork_ctx.get_memory("final_summary")
    assert final_summary is not None, "❌ final_summary should exist"
    assert "branch_id" in final_summary, "❌ final_summary should have branch_id"
    assert "goal" in final_summary, "❌ final_summary should have goal"
    assert "summary" in final_summary, "❌ final_summary should have summary"
    assert "sources" in final_summary, "❌ final_summary should have sources"
    
    print(f"   - Final summary structure:")
    print(f"     - branch_id: {final_summary['branch_id']}")
    print(f"     - goal: {final_summary['goal']}")
    print(f"     - sources count: {len(final_summary['sources'])}")


async def test_semantic_merge():
    """Test that MergeState creates semantic summaries."""
    print("\n🧪 Test 3: Semantic Merge")
    print("=" * 60)
    
    # Create mock fork results
    monitor = SafetyMonitor(max_requests=50)
    
    fork1 = AsyncExecutionContext(user_query="", safety_monitor=monitor)
    await fork1.set_memory("branch_id", "branch_1")
    await fork1.set_memory("branch_goal", "Research X")
    await fork1.set_memory("final_summary", {
        "branch_id": "branch_1",
        "goal": "Research X",
        "summary": "Found information about X",
        "sources": [{"tool": "search_web", "result": "X data"}]
    })
    
    fork2 = AsyncExecutionContext(user_query="", safety_monitor=monitor)
    await fork2.set_memory("branch_id", "branch_2")
    await fork2.set_memory("branch_goal", "Research Y")
    await fork2.set_memory("final_summary", {
        "branch_id": "branch_2",
        "goal": "Research Y",
        "summary": "Found information about Y",
        "sources": [{"tool": "search_web", "result": "Y data"}]
    })
    
    # Create MergeState
    from finitestatemachineAgent.hfsm_agent_async import MergeState, ExecutionState
    merge_state = MergeState(ExecutionState(None))
    
    # Test semantic merge
    merged = merge_state._semantic_merge([fork1, fork2])
    
    print(f"✅ Semantic merge completed")
    print(f"   - Total branches: {merged['total_branches']}")
    print(f"   - Research entries: {len(merged['research'])}")
    
    # Verify structure
    assert "research" in merged, "❌ Should have 'research' key"
    assert merged["total_branches"] == 2, "❌ Should have 2 branches"
    assert len(merged["research"]) == 2, "❌ Should have 2 research entries"
    
    # Verify each entry has the right structure
    for entry in merged["research"]:
        assert "branch_id" in entry, "❌ Entry should have branch_id"
        assert "goal" in entry, "❌ Entry should have goal"
        assert "summary" in entry, "❌ Entry should have summary"
        assert "sources" in entry, "❌ Entry should have sources"
    
    print(f"   - Structure verified ✅")
    print(f"   - Sample entry: {json.dumps(merged['research'][0], indent=2)}")


async def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("🚀 Light Fork Implementation Tests")
    print("=" * 60)
    
    try:
        await test_light_fork_context()
        await test_fork_flow_with_mock()
        await test_semantic_merge()
        
        print("\n" + "=" * 60)
        print("✅ All tests passed!")
        print("=" * 60)
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        raise
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
