"""
Async HFSM Agent
================

Async version of the Hierarchical Finite State Machine Agent.
All states and transitions are async for better concurrency.
"""

from __future__ import annotations

import json
import logging
import asyncio
from datetime import datetime
from abc import ABC, abstractmethod
from typing import AsyncIterator, Optional, List, Dict, Any

from core.context_async import AsyncExecutionContext
from core.executor_async import AsyncToolExecutor
from finitestatemachineAgent.transition import Transition

# Setup logging
logger = logging.getLogger("AsyncAgentEngine")
logger.setLevel(logging.INFO)
if not logger.handlers:
    # Stream Handler (Console)
    sh = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    
    # File Handler (logs/agent.log)
    import os
    if not os.path.exists("logs"):
        os.makedirs("logs", exist_ok=True)
    fh = logging.FileHandler("logs/agent.log", encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

# =================================================================================================
# Base Classes
# =================================================================================================

class AsyncHierarchicalState(ABC):
    """
    Base class for async states in the Hierarchical FSM.
    Enforces immutability via __slots__ to prevent accidental mutable state.
    """
    __slots__ = ("parent",)
    
    def __init__(self, parent: Optional[AsyncHierarchicalState] = None):
        self.parent = parent

    @abstractmethod
    async def handle(self, context: AsyncExecutionContext) -> Optional['Transition | AsyncHierarchicalState']:
        """
        Process context and return next state or transition.
        Can return:
        - Transition object (recommended for explicit semantics)
        - AsyncHierarchicalState instance (legacy support)
        - None (terminal state)
        """
        pass

    async def on_enter(self, context: AsyncExecutionContext):
        """Hook called when entering state."""
        pass

    async def on_exit(self, context: AsyncExecutionContext):
        """Hook called when exiting state."""
        pass

    def find_state_by_type(self, type_name: str) -> AsyncHierarchicalState:
        """Traverse hierarchy to find state."""
        if self.parent:
            return self.parent.find_state_by_type(type_name)
        raise Exception(f"State provider for {type_name} not found")


# =================================================================================================
# Hierarchy States
# =================================================================================================

class AgentRootState(AsyncHierarchicalState):
    """Root of the state hierarchy."""
    def __init__(self):
        super().__init__(parent=None)
        self.find_state_by_type = None  # Will be set by engine

    async def handle(self, context: AsyncExecutionContext):
        return None


class ContextPolicyState(AsyncHierarchicalState):
    """Middleware for context management."""
    async def handle(self, context: AsyncExecutionContext):
        return None


class ReasoningState(AsyncHierarchicalState):
    """Reasoning layer parent."""
    async def handle(self, context: AsyncExecutionContext):
        return None


class ExecutionState(AsyncHierarchicalState):
    """Execution layer parent."""
    async def handle(self, context: AsyncExecutionContext):
        return None


class RecoveryState(AsyncHierarchicalState):
    """Recovery layer parent."""
    async def handle(self, context: AsyncExecutionContext):
        return None


class TerminalState(AsyncHierarchicalState):
    """Terminal state parent."""
    async def handle(self, context: AsyncExecutionContext):
        return None


# =================================================================================================
# Operational States
# =================================================================================================

class RouterState(AsyncHierarchicalState):
    """
    Async router state - decides whether to call tools or answer.
    """
    def __init__(self, parent, llm, registry, tool_choice=None):
        super().__init__(parent)
        self.llm = llm
        self.registry = registry
        self.tool_choice = tool_choice  # Store tool_choice config

    async def handle(self, context: AsyncExecutionContext):
        logger.info("🧠 [Router] Thinking...")

        # Build messages
        system_instruction = await context.get_memory("system_instruction", "")
        messages = [{"role": "system", "content": system_instruction}]

        # Add chat history
        history = await context.get_memory("chat_history", [])
        if history:
            messages.extend(history)

        messages.append({"role": "user", "content": context.user_query})
        
        # Add previous tool calls to context so LLM knows what was already executed
        if context.tool_calls:
            for call in context.tool_calls:
                # Add assistant message with tool call
                tool_call_id = f"call_{call['tool_name']}_{call.get('iteration', 0)}"
                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": tool_call_id,
                        "type": "function",
                        "function": {
                            "name": call["tool_name"],
                            "arguments": json.dumps(call["arguments"])
                        }
                    }]
                })
                
                # Add tool response
                if call.get("result") is not None:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "name": call["tool_name"],
                        "content": str(call["result"])
                    })

        # Call LLM with tools (async) - Use configured tool_choice
        response = await self.llm.chat_with_tools(
            messages=messages,
            tools=self.registry.to_openai_format(),
            tool_choice=self.tool_choice  # Use configured value
        )

        logger.info(f"📊 [Router] Token usage: {response.get('usage', {})}")
        
        # Track token usage in context
        usage = response.get('usage', {})
        if usage:
            await context.accumulate_usage(usage)
        
        # Debug: Log what the LLM returned
        if response.get("content"):
            logger.debug(f"🔍 [Router] LLM also returned content: {response['content'][:100]}...")

        # Check for tool calls
        if response.get("tool_calls"):
            logger.info(f"🔧 [Router] {len(response['tool_calls'])} tool(s) selected")

            # Store tool calls (async)
            for call in response["tool_calls"]:
                await context.add_tool_call(
                    tool_name=call["function"]["name"],
                    arguments=json.loads(call["function"]["arguments"]),
                    result=None
                )

            return self.find_state_by_type("ToolState")
        else:
            logger.warning("[Router] No tool calls generated by LLM.")
            return self.find_state_by_type("AnswerState")


class ToolState(AsyncHierarchicalState):
    """
    Async tool execution state.
    """
    def __init__(self, parent, executor: AsyncToolExecutor, skip_validation: bool = True):
        super().__init__(parent)
        self.executor = executor
        self.skip_validation = skip_validation

    async def handle(self, context: AsyncExecutionContext):
        logger.info("🛠️ [Tool] Executing tools...")

        # Get pending tool calls
        pending = [c for c in context.tool_calls if c.get("result") is None]

        if not pending:
            logger.warning("⚠️ [Tool] No pending tool calls")
            return self.find_state_by_type("AnswerState")

        # Execute all tools concurrently (async)
        results = await self.executor.execute_parallel(pending)

        # Update context atomically using public API
        await context.update_tool_results(pending, results)
        
        for call in pending:
            logger.info(f"✅ [Tool] Result for {call['tool_name']}: {str(call['result'])[:100]}...")

        # Check validation config
        if self.skip_validation:
            logger.info("⏩ [Tool] Skipping validation (configured)")
            return self.find_state_by_type("AnswerState")
        else:
            logger.info("🔍 [Tool] Proceeding to validation")
            return self.find_state_by_type("ValidationState")


class ValidationState(AsyncHierarchicalState):
    """
    Async validation state with customizable validation logic.
    """
    def __init__(self, parent, llm, validation_fn=None):
        super().__init__(parent)
        self.llm = llm
        self.validation_fn = validation_fn  # Custom validation function

    async def handle(self, context: AsyncExecutionContext):
        logger.info("🔍 [Validation] Checking data...")

        # Get last tool call atomically (thread-safe)
        async with context._lock:
            if not context.tool_calls:
                logger.warning("⚠️ [Validation] No tool calls to validate")
                return self.find_state_by_type("AnswerState")
            
            last_call = context.tool_calls[-1]
            tool_name = last_call.get("tool_name")
            result = last_call.get("result")
        
        # Use custom validation function if provided
        if self.validation_fn:
            is_valid = await self.validation_fn(context, tool_name, result)
            logger.info(f"✅ [Validation] Custom validation result: {is_valid}")
        else:
            # Default: simple LLM-based validation
            messages = [
                {"role": "system", "content": "Validate if the tool results answer the query."},
                {"role": "user", "content": f"Query: {context.user_query}"},
                {"role": "user", "content": f"Results: {json.dumps(result)}"}
            ]

            response = await self.llm.chat(messages)
            is_valid = "true" in response["content"].lower()
            logger.info(f"✅ [Validation] Default validation result: {is_valid}")

        if is_valid:
            return self.find_state_by_type("AnswerState")
        else:
            return self.find_state_by_type("RetryState")


class AnswerState(TerminalState):
    """
    Async answer state with streaming.
    """
    def __init__(self, parent, llm):
        super().__init__(parent)
        self.llm = llm

    async def handle(self, context: AsyncExecutionContext):
        logger.info("✅ [Answer] Generating final response...")

        # Build messages
        system_instruction = await context.get_memory("system_instruction", "")
        messages = [{"role": "system", "content": system_instruction}]

        # Add history
        history = await context.get_memory("chat_history", [])
        if history:
            messages.extend(history)

        messages.append({"role": "user", "content": context.user_query})

        # Add tool results if any
        if context.tool_calls:
            for call in context.tool_calls:
                if call.get("result"):
                    tool_call_id = f"call_{call['tool_name']}_{call.get('iteration', 0)}"
                    messages.append({
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{
                            "id": tool_call_id,
                            "type": "function",
                            "function": {
                                "name": call["tool_name"],
                                "arguments": json.dumps(call["arguments"])
                            }
                        }]
                    })
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "name": call["tool_name"],
                        "content": str(call.get("result", ""))
                    })

            messages.append({
                "role": "system",
                "content": "Based on the tool results, provide a best-effort answer. If the information is insufficient or invalid, explain clearly what is missing. Do NOT call more tools."
            })

        # Create async streaming generator and store in context
        logger.info("🌊 [Answer] Streaming initialized")
        generator = self.llm.chat_stream(messages, context)
        await context.set_memory("answer_stream", generator)
        return None  # Terminal state


class RetryState(AsyncHierarchicalState):
    """
    Async retry state.
    """
    def __init__(self, parent):
        super().__init__(parent)

    async def handle(self, context: AsyncExecutionContext):
        logger.warning("⚠️ [Retry] Attempting recovery...")

        await context.increment_iteration()

        if context.current_iteration >= context.max_iterations:
            logger.warning("⚠️ [Retry] Max retries reached. Proceeding to AnswerState (Best Effort).")
            return self.find_state_by_type("AnswerState")
        else:
            logger.warning(f"⚠️ [Retry] Attempt {context.current_iteration}/{context.max_iterations}")
            return self.find_state_by_type("RouterState")


class FailState(TerminalState):
    """
    Async fail state.
    """
    def __init__(self, parent):
        super().__init__(parent)

    async def handle(self, context: AsyncExecutionContext):
        logger.error("❌ [Fail] Terminating agent.")
        return None


# =================================================================================================
# Agent Engine
# =================================================================================================

class AsyncAgentEngine:
    """
    Async Hierarchical Finite State Machine Agent Engine.
    """

    def __init__(
        self,
        llm,
        registry,
        executor: AsyncToolExecutor,
        system_instruction: str = "",
        tool_choice: Optional[str] = None,
        skip_validation: bool = True,  # Default: No validation
        validation_fn=None  # Custom validation function
    ):
        self.llm = llm
        self.registry = registry
        self.executor = executor
        self.system_instruction = system_instruction
        self.tool_choice = tool_choice
        self.skip_validation = skip_validation
        self.validation_fn = validation_fn

        self.states = {}

        # Initialize hierarchy
        self.root = AgentRootState()
        self.states["AgentRootState"] = self.root

        self.policy = ContextPolicyState(self.root)
        self.states["ContextPolicyState"] = self.policy

        self.reasoning = ReasoningState(self.policy)
        self.states["ReasoningState"] = self.reasoning

        self.execution = ExecutionState(self.policy)
        self.states["ExecutionState"] = self.execution

        self.recovery = RecoveryState(self.policy)
        self.states["RecoveryState"] = self.recovery

        self.terminal = TerminalState(self.policy)
        self.states["TerminalState"] = self.terminal

        # Operational states
        self.router_state = RouterState(self.reasoning, self.llm, self.registry, self.tool_choice)
        self.states["RouterState"] = self.router_state

        self.tool_state = ToolState(self.execution, self.executor, self.skip_validation)
        self.states["ToolState"] = self.tool_state

        self.validation_state = ValidationState(self.execution, self.llm, self.validation_fn)
        self.states["ValidationState"] = self.validation_state

        self.answer_state = AnswerState(self.reasoning, self.llm)
        self.states["AnswerState"] = self.answer_state

        self.retry_state = RetryState(self.execution)
        self.states["RetryState"] = self.retry_state

        self.fail_state = FailState(self.root)
        self.states["FailState"] = self.fail_state

        # Service locator
        self.root.find_state_by_type = self._find_state_provider

    def _find_state_provider(self, state_type):
        """Find state by type."""
        for state in self.states.values():
            if type(state).__name__ == state_type:
                return state
        raise ValueError(f"State {state_type} not found")
    
    def _save_snapshot(self, context: AsyncExecutionContext, event_name: str):
        """Save context snapshot to disk/log."""
        try:
            snapshot = context.snapshot()
            
            # Disk Persistence
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"snapshot_{timestamp}_{event_name}.json"
            directory = "logs/snapshots"
            
            import os
            if not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
                
            filepath = os.path.join(directory, filename)
            
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(snapshot, f, ensure_ascii=False, indent=2)
                
            logger.debug(f"💾 Snapshot saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save snapshot: {e}")

    async def dispatch(self, context: AsyncExecutionContext):
        """
        Async state machine dispatch loop with transition resolution.
        """
        current_state = self.router_state

        while current_state:
            state_name = type(current_state).__name__
            logger.info(f"📍 [Engine] Current state: {state_name}")

            # Handle state (async)
            result = await current_state.handle(context)

            if result is None:
                logger.info(f"🏁 [Engine] Reached terminal state: {state_name}")
                break

            # Resolve transition
            if isinstance(result, Transition):
                next_state = self._find_state_provider(result.to)
                logger.info(f"🔄 Transition: {state_name} -> {result.to} (reason: {result.reason})")
                if result.metadata:
                    logger.debug(f"   Metadata: {result.metadata}")
            else:
                # Legacy: direct state return
                next_state = result
                next_name = type(next_state).__name__
                logger.info(f"🔄 Transition: {state_name} -> {next_name}")

            # Save snapshot
            self._save_snapshot(context, f"transition_{state_name}_to_{type(next_state).__name__}")

            current_state = next_state

    async def run(self, query: str, chat_history=None) -> AsyncExecutionContext:
        """
        Run agent (async).
        """
        context = AsyncExecutionContext(user_query=query)
        await context.set_memory("system_instruction", self.system_instruction)
        await context.set_memory("chat_history", chat_history or [])

        await self.dispatch(context)

        return context

    async def run_stream(self, query: str, chat_history=None) -> AsyncIterator[str]:
        """
        Run with streaming (async generator).
        """
        context = await self.run(query, chat_history)

        # Stream from context memory (not from state instance)
        stream = await context.get_memory("answer_stream")
        if stream:
            async for token in stream:
                yield token
