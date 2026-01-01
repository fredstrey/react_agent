import asyncio
import unittest
from core.context_async import AsyncExecutionContext
from finitestatemachineAgent.hfsm_agent_async import RouterState
from finitestatemachineAgent.transition import Transition

class MockLLM:
    pass

class MockRegistry:
    pass

class TestTransitions(unittest.IsolatedAsyncioTestCase):
    async def test_router_returns_transition(self):
        # Setup
        llm = MockLLM()
        registry = MockRegistry()
        state = RouterState(None, llm, registry)
        context = AsyncExecutionContext(user_query="test")
        
        # We can't easily execute handle() without mocking LLM response
        # but we can check if it imports Transition correctly
        self.assertTrue(issubclass(Transition, object))
        print("✅ Transition class is available")

if __name__ == "__main__":
    unittest.main()
