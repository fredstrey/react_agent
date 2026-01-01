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
from typing import AsyncIterator, Optional, List, Dict, Any, Literal

from pydantic import BaseModel, Field
from core.context_async import AsyncExecutionContext, SafetyMonitor
from core.executor_async import AsyncToolExecutor
from finitestatemachineAgent.transition import Transition
from finitestatemachineAgent.fork_states import ResearchForkState, ForkSummaryState

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
# Domain Models for Parallel Execution
# =================================================================================================

class BranchSpec(BaseModel):
    """Specification for a parallel execution branch."""
    id: str
    goal: str
    constraints: List[str] = Field(default_factory=list)

class ParallelPlan(BaseModel):
    """Plan for parallel execution strategy."""
    strategy: Literal["single", "parallel_research"]
    branches: List[BranchSpec] = Field(default_factory=list)
    merge_policy: str = "append"

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
    def __init__(self, parent, llm, registry, tool_choice=None, enable_parallel=False):
        super().__init__(parent)
        self.llm = llm
        self.registry = registry
        self.tool_choice = tool_choice  # Store tool_choice config
        self.enable_parallel = enable_parallel  # Store parallel execution flag

    async def handle(self, context: AsyncExecutionContext):
        logger.info("🧠 [Router] Thinking...")
        
        # 🔥 REMOVED: Old merged_context injection (now handled in AnswerState)
        # Research context is now injected ONLY in AnswerState for better control

        # 🔥 NEW: Check for parallel planning BEFORE calling LLM
        # Only if enabled, not checked yet, AND IS ROOT CONTEXT (no parents allowed to fork)
        parallel_checked = await context.get_memory("parallel_checked", False)
        is_root = context.parent is None
        
        if (self.enable_parallel and not parallel_checked and is_root):
            
            # Mark as checked so we don't loop back here endlessly
            await context.set_memory("parallel_checked", True)
            return Transition(to="ParallelPlanningState", reason="Parallel planning check required")

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

        # 🛡️ Safety Check Before LLM Call
        await context.increment_llm_call()

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

            # Proceed to tool execution
            return Transition(to="ToolState", reason="Tools selected by LLM", metadata={"tool_count": len(response["tool_calls"])})
        else:
            logger.warning("[Router] No tool calls generated by LLM.")
            return Transition(to="AnswerState", reason="Direct answer generation")


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
            return Transition(to="AnswerState", reason="No pending tool calls")

        # Execute all tools concurrently (async)
        results = await self.executor.execute_parallel(pending)

        # Update context atomically using public API
        await context.update_tool_results(pending, results)
        
        for call in pending:
            logger.info(f"✅ [Tool] Result for {call['tool_name']}: {str(call['result'])[:100]}...")

        # Check validation config
        if self.skip_validation:
            logger.info("⏩ [Tool] Skipping validation (configured)")
            
            # 🔥 NEW: Check if we're in a fork context
            branch_id = await context.get_memory("branch_id")
            if branch_id:
                # Fork flow: go to ForkSummaryState
                logger.info(f"🌿 [Tool] Fork context detected, proceeding to summary")
                return Transition(to="ForkSummaryState", reason="Fork tool execution complete")
            else:
                # Normal flow: go to AnswerState
                return Transition(to="AnswerState", reason="Validation skipped by config")
        else:
            logger.info("🔍 [Tool] Proceeding to validation")
            return Transition(to="ValidationState", reason="Validation required")


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
                return Transition(to="AnswerState", reason="No tools to validate")
            
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

            # 🛡️ Safety Check Before LLM Call
            await context.increment_llm_call()

            response = await self.llm.chat(messages)
            is_valid = "true" in response["content"].lower()
            logger.info(f"✅ [Validation] Default validation result: {is_valid}")

        if is_valid:
            return Transition(to="AnswerState", reason="Validation passed")
        else:
            return Transition(to="RetryState", reason="Validation failed", metadata={"tool": tool_name})


class ParallelPlanningState(AsyncHierarchicalState):
    """
    Async parallel planning state with customizable planning logic.
    Decides whether to execute tools in parallel or sequentially.
    """
    def __init__(self, parent, llm, planner_fn=None, planning_system_prompt=None, max_branches=3):
        super().__init__(parent)
        self.llm = llm
        self.planner_fn = planner_fn  # Custom planner function
        self.planning_system_prompt = planning_system_prompt  # Custom/enhanced system prompt
        self.max_branches = max_branches  # 🔥 NEW: Max branches allowed
    

    
    async def _default_llm_plan(self, context: AsyncExecutionContext) -> ParallelPlan:
        """Default LLM-based planning logic with customizable system prompt."""
        
        # Default system prompt (NO placeholders)
        default_prompt = """You are a planning module for a tool execution system.

Analyze the user's request and decide if it should be split into independent parallel research branches.

Rules:
- Branches must be INDEPENDENT (no dependencies between them)
- Each branch should research a different aspect
- Only use parallel if it provides clear value
- Do NOT plan tool execution, only research goals
- MAX BRANCHES ALLOWED: {max_branches} (if you need more, prioritize top {max_branches}) 

If parallel is NOT useful, respond with:
{{"strategy": "single"}}

If parallel IS useful, respond with:
{{
  "strategy": "parallel_research",
  "branches": [
    {{"id": "branch_1", "goal": "Research X", "constraints": []}},
    {{"id": "branch_2", "goal": "Research Y", "constraints": []}}
  ],
  "merge_policy": "append"
}}

Respond with valid JSON only."""

        # Determine final system prompt
        if self.planning_system_prompt is None:
            # Use default system prompt formatted with limit
            system_prompt = default_prompt.format(max_branches=self.max_branches)
        elif callable(self.planning_system_prompt):
            # Callable: incremental enhancement
            # If the user uses a callable, we assume they WANT the base prompt instructions
            # so we format the default prompt first
            base_prompt = default_prompt.format(max_branches=self.max_branches)
            system_prompt = self.planning_system_prompt(base_prompt, context)
        else:
            # String: complete override
            # If user overrides, they are responsible for limits in instructions, 
            # but we will still enforce in code.
            system_prompt = self.planning_system_prompt

        # Construct messages properly (avoid str.format on JSON braces)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"User query: {context.user_query}"}
        ]
        
        try:
            # 🛡️ Safety Check Before LLM Call
            await context.increment_llm_call()
            
            response = await self.llm.chat(messages)
            content = response["content"]
            
            # Extract JSON if wrapped in markdown code blocks
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
                
            plan_data = json.loads(content.strip())
            
            # Parse into ParallelPlan
            if plan_data.get("strategy") == "parallel_research":
                branches = [
                    BranchSpec(
                        id=b["id"],
                        goal=b["goal"],
                        constraints=b.get("constraints", [])
                    )
                    for b in plan_data.get("branches", [])
                ]
                return ParallelPlan(
                    strategy="parallel_research",
                    branches=branches,
                    merge_policy=plan_data.get("merge_policy", "append")
                )
            else:
                return ParallelPlan(strategy="single")
        
        except Exception as e:
            logger.error(f"❌ [ParallelPlan] Planning failed: {e}")
            return ParallelPlan(strategy="single")  # Safe fallback
    
    async def handle(self, context: AsyncExecutionContext):
        logger.info("🧩 [ParallelPlan] Evaluating parallelization strategy...")
        
        # Use custom planner if provided
        if self.planner_fn:
            plan = await self.planner_fn(context)
            logger.info(f"✅ [ParallelPlan] Custom planner result: {plan.strategy}")
        else:
            # Default: LLM-based planning
            plan = await self._default_llm_plan(context)
            logger.info(f"✅ [ParallelPlan] LLM planner result: {plan.strategy}")
        
        # Validate plan
        if not isinstance(plan, ParallelPlan):
            logger.warning("⚠️ [ParallelPlan] Invalid plan type, falling back to single")
            return Transition(to="RouterState", reason="Invalid plan type", metadata={"fallback": True})
        
        # Store plan in context
        # Use model_dump() for Pydantic serialization safety
        await context.set_memory("parallel_plan", plan.model_dump())
        
        # Decide next state based on strategy
        if plan.strategy == "parallel_research" and plan.branches:
            logger.info(f"🔀 [ParallelPlan] Parallel execution with {len(plan.branches)} branches")
            return Transition(to="ForkDispatchState", reason="Parallel execution strategy selected", metadata={"branches": len(plan.branches)})
        else:
            logger.info("➡️ [ParallelPlan] Single path execution")
            # Mark as checked ensures Router continues
            return Transition(to="RouterState", reason="Single path strategy selected")


class ForkDispatchState(AsyncHierarchicalState):
    """
    Dispatches parallel execution branches using context forking.
    Each branch runs independently with its own context.
    """
    def __init__(self, parent, engine):
        super().__init__(parent)
        self.engine = engine  # Reference to engine for dispatch
    
    async def handle(self, context: AsyncExecutionContext):
        logger.info("🔀 [ForkDispatch] Starting parallel execution...")
        
        # Get plan from context
        plan_data = await context.get_memory("parallel_plan")
        if not plan_data:
            logger.warning("⚠️ [ForkDispatch] No valid plan, falling back")
            return Transition(to="RouterState", reason="Missing parallel plan")
        
        # Handle both dict (serialized) and object (in-memory) cases
        if isinstance(plan_data, dict):
            try:
                plan = ParallelPlan(**plan_data)
            except Exception as e:
                logger.error(f"❌ [ForkDispatch] Invalid plan data: {e}")
                return Transition(to="RouterState", reason="Invalid plan data")
        elif isinstance(plan_data, ParallelPlan):
            plan = plan_data
        else:
            logger.error(f"❌ [ForkDispatch] Unknown plan type: {type(plan_data)}")
            return Transition(to="RouterState", reason="Unknown plan type")
            
        if not plan.branches:
            logger.warning("⚠️ [ForkDispatch] No branches in plan, falling back")
            return Transition(to="RouterState", reason="Empty branches in plan")
        
        logger.info(f"🔀 [ForkDispatch] Spawning {len(plan.branches)} branches")
        
        # Create forked contexts
        fork_contexts = []
        for branch in plan.branches:
            fork_ctx = context.fork()
            await fork_ctx.set_memory("branch_id", branch.id)
            await fork_ctx.set_memory("branch_goal", branch.goal)
            await fork_ctx.set_memory("branch_constraints", branch.constraints)
            
            # Increment recursion depth
            current_depth = await context.get_memory("parallel_depth", 0)
            await fork_ctx.set_memory("parallel_depth", current_depth + 1)
            
            # Override user query with branch goal
            fork_ctx.user_query = f"{context.user_query}\n\nBranch goal: {branch.goal}"
            
            # 🔥 NEW: Detailed logging for observability
            logger.info(f"🌿 [ForkDispatch] Creating fork '{branch.id}':")
            logger.info(f"   📋 Task: {branch.goal}")
            logger.info(f"   📝 Query: {fork_ctx.user_query[:200]}..." if len(fork_ctx.user_query) > 200 else f"   📝 Query: {fork_ctx.user_query}")
            if branch.constraints:
                logger.info(f"   ⚠️  Constraints: {', '.join(branch.constraints)}")
            
            fork_contexts.append((branch.id, fork_ctx))
        
        # Execute all forks in parallel
        try:
            results = await asyncio.gather(
                *[self._execute_fork(branch_id, fork_ctx) for branch_id, fork_ctx in fork_contexts],
                return_exceptions=True
            )
            
            # Filter out failed forks
            successful_forks = []
            for (branch_id, fork_ctx), result in zip(fork_contexts, results):
                if isinstance(result, Exception):
                    logger.error(f"❌ [ForkDispatch] Branch {branch_id} failed: {result}")
                else:
                    successful_forks.append(fork_ctx)
                    logger.info(f"✅ [ForkDispatch] Branch {branch_id} completed")
            
            # 🔥 NEW: Accumulate token usage from all forks back to parent
            total_fork_tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            for fork_ctx in successful_forks:
                fork_usage = await fork_ctx.get_total_usage()
                total_fork_tokens["prompt_tokens"] += fork_usage.get("prompt_tokens", 0)
                total_fork_tokens["completion_tokens"] += fork_usage.get("completion_tokens", 0)
                total_fork_tokens["total_tokens"] += fork_usage.get("total_tokens", 0)
            
            if total_fork_tokens["total_tokens"] > 0:
                await context.accumulate_usage(total_fork_tokens)
                logger.info(f"📊 [ForkDispatch] Accumulated {total_fork_tokens['total_tokens']} tokens from {len(successful_forks)} fork(s)")
            
            # Store results
            await context.set_memory("fork_results", successful_forks)
            
            if successful_forks:
                return Transition(to="MergeState", reason="Forks completed", metadata={"successful_forks": len(successful_forks)})
            else:
                logger.warning("⚠️ [ForkDispatch] All forks failed, falling back")
                return Transition(to="RouterState", reason="All forks failed")
        
        except Exception as e:
            logger.error(f"❌ [ForkDispatch] Parallel execution failed: {e}")
            return Transition(to="RouterState", reason="Parallel execution exception", metadata={"error": str(e)})
    
    async def _execute_fork(self, branch_id: str, fork_ctx: AsyncExecutionContext):
        """Execute a single fork through the engine."""
        logger.info(f"🌿 [Fork:{branch_id}] Starting execution")
        
        # 🔥 NEW: Start from ResearchForkState (not RouterState)
        # This bypasses the full Router logic for efficiency
        initial_state = self.find_state_by_type("ResearchForkState")
        await self.engine.dispatch(fork_ctx)
        
        logger.info(f"🌿 [Fork:{branch_id}] Execution complete")
        return fork_ctx


class MergeState(AsyncHierarchicalState):
    """
    Merges results from parallel fork executions.
    Supports customizable merge strategies.
    """
    def __init__(self, parent, merge_fn=None):
        super().__init__(parent)
        self.merge_fn = merge_fn  # Custom merge function
    
    async def handle(self, context: AsyncExecutionContext):
        logger.info("🧬 [Merge] Consolidating fork results...")
        
        # Get fork results
        fork_results = await context.get_memory("fork_results", [])
        
        if not fork_results:
            logger.warning("⚠️ [Merge] No fork results to merge")
            return Transition(to="RouterState", reason="No results to merge")
        
        logger.info(f"🧬 [Merge] Merging {len(fork_results)} fork results")
        
        # Use custom merge function if provided
        if self.merge_fn:
            try:
                plan_data = await context.get_memory("parallel_plan")
                plan = None
                if isinstance(plan_data, dict):
                    plan = ParallelPlan(**plan_data)
                elif isinstance(plan_data, ParallelPlan):
                    plan = plan_data
                    
                merged = await self.merge_fn(context, fork_results, plan)
                logger.info("✅ [Merge] Custom merge completed")
            except Exception as e:
                logger.error(f"❌ [Merge] Custom merge failed: {e}, using default")
                merged = self._semantic_merge(fork_results)
        else:
            # 🔥 NEW: Use semantic merge by default
            merged = self._semantic_merge(fork_results)
            logger.info("✅ [Merge] Semantic merge completed")
        
        # Store merged results as research_context (used by AnswerState)
        await context.set_memory("research_context", merged)
        
        # Continue to RouterState (which will then go to AnswerState)
        return Transition(to="RouterState", reason="Research context merged", metadata={"branches": len(fork_results)})
    
    def _semantic_merge(self, fork_results: List[AsyncExecutionContext]) -> dict:
        """Semantic merge strategy: extract structured summaries from forks."""
        research = []
        
        for fork_ctx in fork_results:
            # Get the final_summary created by ForkSummaryState
            final_summary = fork_ctx.memory.get("final_summary")
            
            if final_summary:
                # Use the structured summary
                research.append(final_summary)
            else:
                # Fallback: construct from available data
                branch_id = fork_ctx.memory.get("branch_id", "unknown")
                branch_goal = fork_ctx.memory.get("branch_goal", "")
                
                # Extract tool results
                sources = []
                for call in fork_ctx.tool_calls:
                    sources.append({
                        "tool": call["tool_name"],
                        "result": str(call.get("result", ""))[:300]
                    })
                
                research.append({
                    "branch_id": branch_id,
                    "goal": branch_goal,
                    "summary": f"Executed {len(sources)} tool(s)",
                    "sources": sources
                })
        
        return {
            "research": research,
            "total_branches": len(research)
        }


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

        # 🔥 NEW: Inject research context from parallel execution (if exists)
        research_context = await context.get_memory("research_context")
        if research_context:
            logger.info(f"📚 [Answer] Injecting research from {research_context.get('total_branches', 0)} branches")
            messages.append({
                "role": "system",
                "content": "Use the following research results to answer the user's question:"
            })
            messages.append({
                "role": "user",
                "content": json.dumps(research_context, ensure_ascii=False, indent=2)
            })

        # Create async streaming generator and store in context
        logger.info("🌊 [Answer] Streaming initialized")
        
        # 🛡️ Safety Check Before LLM Call (Streaming)
        await context.increment_llm_call()
        
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
            return Transition(to="AnswerState", reason="Max retries reached")
        else:
            logger.warning(f"⚠️ [Retry] Attempt {context.current_iteration}/{context.max_iterations}")
            return Transition(to="RouterState", reason="Retrying execution", metadata={"attempt": context.current_iteration})


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
        validation_fn=None,  # Custom validation function
        
        # 🔥 NEW: Parallel execution support
        enable_parallel_planning: bool = False,
        parallel_plan_fn=None,           # Custom planning function
        planning_system_prompt=None,     # Custom/enhanced system prompt for LLM planner
        merge_fn=None,                   # Custom merge strategy
        max_parallel_branches: int = 3,  # 🔥 NEW: Max branches per fork (width limit)
        # 🔥 Safety Config
        max_global_requests: int = 50
    ):
        self.llm = llm
        self.registry = registry
        self.executor = executor
        self.system_instruction = system_instruction
        self.tool_choice = tool_choice
        self.skip_validation = skip_validation
        self.validation_fn = validation_fn
        
        # Store parallel execution config
        self.enable_parallel_planning = enable_parallel_planning
        self.parallel_plan_fn = parallel_plan_fn
        self.planning_system_prompt = planning_system_prompt
        self.merge_fn = merge_fn
        self.max_parallel_branches = max_parallel_branches
        
        # Store safety config
        self.max_global_requests = max_global_requests

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
        self.router_state = RouterState(
            self.reasoning, 
            self.llm, 
            self.registry, 
            self.tool_choice,
            enable_parallel=self.enable_parallel_planning  # 🔥 Fix: Pass parallel flag
        )
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
        
        # Add parallel execution states (only if enabled)
        if self.enable_parallel_planning:
            self.parallel_planning_state = ParallelPlanningState(
                self.reasoning, 
                self.llm, 
                self.parallel_plan_fn,
                self.planning_system_prompt,
                max_branches=self.max_parallel_branches  # 🔥 Fix: Pass width limit
            )
            self.states["ParallelPlanningState"] = self.parallel_planning_state
            
            self.fork_dispatch_state = ForkDispatchState(self.execution, self)
            self.states["ForkDispatchState"] = self.fork_dispatch_state
            
            self.merge_state = MergeState(self.execution, self.merge_fn)
            self.states["MergeState"] = self.merge_state
            
            # 🔥 NEW: Fork-specific states
            self.research_fork_state = ResearchForkState(
                self.reasoning,
                self.llm,
                self.registry,
                self.tool_choice
            )
            self.states["ResearchForkState"] = self.research_fork_state
            
            self.fork_summary_state = ForkSummaryState(self.terminal, self.llm)
            self.states["ForkSummaryState"] = self.fork_summary_state
            
            logger.info("✅ Parallel execution states initialized")

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
        """
        Run agent (async).
        """
        # Create context with safety monitor
        monitor = SafetyMonitor(max_requests=self.max_global_requests)
        context = AsyncExecutionContext(user_query=query, safety_monitor=monitor)
        
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
