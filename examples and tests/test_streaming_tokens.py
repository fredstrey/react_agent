"""
Simple Test for Streaming Token Accumulation
==============================================

Direct test without complex mocking.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
from core.context_async import AsyncExecutionContext, SafetyMonitor


async def simple_stream_with_usage():
    """Simple async generator that yields tokens and usage."""
    yield "Hello"
    yield " world"
    yield {"__usage__": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}}


async def test_usage_accumulation():
    """Test that usage dict is detected and accumulated."""
    print("\n🧪 Test: Usage Accumulation from Stream")
    print("=" * 60)
    
    # Create context
    monitor = SafetyMonitor(max_requests=50)
    context = AsyncExecutionContext(user_query="Test", safety_monitor=monitor)
    
    # Initial usage
    await context.accumulate_usage({"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8})
    print(f"📊 Initial usage: {await context.get_total_usage()}")
    
    # Simulate streaming with usage accumulation
    full_response = ""
    async for token in simple_stream_with_usage():
        if isinstance(token, dict) and "__usage__" in token:
            # Accumulate usage
            await context.accumulate_usage(token["__usage__"])
            print(f"📊 Accumulated streaming usage: {token['__usage__']}")
        else:
            # Regular token
            full_response += token
    
    print(f"📝 Response: '{full_response}'")
    
    # Check final usage
    final_usage = await context.get_total_usage()
    print(f"📊 Final usage: {final_usage}")
    
    # Verify
    expected_total = 8 + 12  # Initial + streaming
    assert final_usage["total_tokens"] == expected_total, \
        f"❌ Expected {expected_total}, got {final_usage['total_tokens']}"
    
    print(f"✅ Usage accumulation works correctly!")
    print(f"   - Initial: 8 tokens")
    print(f"   - Streaming: 12 tokens")
    print(f"   - Final: {final_usage['total_tokens']} tokens")


async def main():
    """Run test."""
    print("\n" + "=" * 60)
    print("🚀 Simple Streaming Token Test")
    print("=" * 60)
    
    try:
        await test_usage_accumulation()
        
        print("\n" + "=" * 60)
        print("✅ Test passed!")
        print("=" * 60)
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
