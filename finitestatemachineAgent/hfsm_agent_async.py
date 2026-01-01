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
    def __init__(self, parent, llm, registry, tool_choice=None, enable_parallel=False, post_hook=None):
        super().__init__(parent)
        self.llm = llm
        self.registry = registry
        self.tool_choice = tool_choice  # Store tool_choice config
        self.enable_parallel = enable_parallel  # Store parallel execution flag
        self.post_hook = post_hook  # 🔥 Domain-specific hook

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
        # 🔥 Optimization: Skip history if intent was analyzed (query is already enhanced)
        # or if this is a fork (context pollution)
        intent_analyzed = await context.get_memory("intent_analyzed", False)
        is_root = context.parent is None
        
        should_include_history = is_root and not intent_analyzed
        
        history = await context.get_memory("chat_history", [])
        if history and should_include_history:
            logger.info(f"📜 [Router] Including {len(history)} history messages")
            messages.extend(history)
        elif history:
            logger.info("🧹 [Router] Skipping history (Intent analyzed or Fork)")

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
            transition = Transition(to="ToolState", reason="Tools selected by LLM", metadata={"tool_count": len(response["tool_calls"])})
        else:
            logger.warning("[Router] No tool calls generated by LLM.")
            transition = Transition(to="AnswerState", reason="Direct answer generation")
        
        # 🔥 Apply post-hook if provided (domain-specific logic)
        if self.post_hook:
            transition = await self.post_hook(context, transition)
        
        return transition


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

        # Decide next state based on validation config
        if self.skip_validation:
            logger.info("⏩ [Tool] Skipping validation (configured)")
            
            # 🔥 CRITICAL: Check if we're in a fork context
            branch_id = await context.get_memory("branch_id")
            if branch_id:
                # We're in a fork - go to ForkContractState
                logger.info(f"🌿 [Tool] Fork context detected (branch: {branch_id}), proceeding to contract extraction")
                return Transition(to="ForkContractState", reason="Fork execution complete")
            else:
                # Normal flow - go to AnswerState
                return Transition(to="AnswerState", reason="Tools executed, validation skipped")
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

    async def _validate(self, context: AsyncExecutionContext, tool_name: str, result: any) -> bool:
        """Internal validation logic."""
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
        
        return is_valid

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
        
        # Validate
        is_valid = await self._validate(context, tool_name, result)
        
        if is_valid:
            logger.info("✅ [Validation] Data is valid")
            
            # 🔥 CRITICAL: Check if we're in a fork context
            branch_id = await context.get_memory("branch_id")
            if branch_id:
                # We're in a fork - go to ForkContractState
                logger.info(f"🌿 [Validation] Fork context detected (branch: {branch_id}), proceeding to contract extraction")
                return Transition(to="ForkContractState", reason="Validation passed in fork")
            else:
                # Normal flow - go to AnswerState
                return Transition(to="AnswerState", reason="Validation passed")
        else:
            logger.warning("⚠️ [Validation] Data is invalid")
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
                merged = self._contract_merge(fork_results)
        else:
            # 🔥 NEW: Use deterministic contract merge by default
            merged = self._contract_merge(fork_results)
            logger.info("✅ [Merge] Contract merge completed")
        
        # Store merged results as research_context (used by SynthesisState or AnswerState)
        await context.set_memory("research_context", merged)
        
        # 🔥 CRITICAL: Route based on whether synthesis is needed
        # Synthesis is only needed when we have fork results (parallel execution)
        fork_results = await context.get_memory("fork_results", [])
        
        if fork_results and len(fork_results) > 0:
            # We have fork results - go to synthesis
            logger.info(f"🧬 [Merge] Routing to synthesis ({len(fork_results)} forks)")
            return Transition(to="SemanticSynthesisState", reason="Contracts merged, ready for synthesis", metadata={"branches": len(fork_results)})
        else:
            # No forks - skip synthesis, go directly to router
            logger.info("🔄 [Merge] No forks, skipping synthesis")
            return Transition(to="RouterState", reason="No synthesis needed", metadata={"branches": 0})
    
    def _contract_merge(self, fork_results: List[AsyncExecutionContext]) -> dict:
        """
        Deterministic contract merge (no LLM, domain-agnostic).
        
        Detects consensus and conflicts in claims from multiple forks.
        """
        from finitestatemachineAgent.fork_contracts import ForkResult, MergedContract, UncertainTopic
        
        all_claims = {}
        all_coverage = set()
        all_uncertain = []  # List of UncertainTopic dicts
        
        # Collect all claims by key
        for fork_ctx in fork_results:
            contract_data = fork_ctx.memory.get("fork_contract")
            if not contract_data:
                logger.warning(f"⚠️ [Merge] Fork {fork_ctx.memory.get('branch_id')} has no contract")
                continue
            
            # 🔥 DEBUG: Log each fork's contract
            logger.info(f"📦 [Merge] Processing fork {fork_ctx.memory.get('branch_id')}:")
            logger.info(f"   Contract keys: {list(contract_data.keys())}")
            logger.info(f"   Claims count: {len(contract_data.get('claims', []))}")
            
            try:
                contract = ForkResult(**contract_data)
                
                # 🔥 DEBUG: Log claims being added
                logger.info(f"   Adding {len(contract.claims)} claims to merge:")
                for claim in contract.claims:
                    logger.info(f"      - {claim.key}: {claim.value}")
                
                for claim in contract.claims:
                    if claim.key not in all_claims:
                        all_claims[claim.key] = []
                    all_claims[claim.key].append({
                        "value": claim.value,
                        "evidence": [ev.model_dump() for ev in claim.evidence],  # Serialize Evidence objects
                        "confidence": claim.confidence,
                        "branch_id": contract.branch_id
                    })
                
                all_coverage.update(contract.coverage)
                
                # 🔥 EPISTEMIC: Collect uncertain topics (don't invalidate claims)
                for uncertain in contract.uncertain_topics:
                    all_uncertain.append(uncertain.model_dump())
                
            except Exception as e:
                logger.error(f"❌ [Merge] Failed to parse fork contract: {e}")
                continue
        
        # Detect consensus and conflicts
        resolved = {}
        conflicts = {}
        
        for key, claim_list in all_claims.items():
            # Extract unique values (convert to string for comparison)
            unique_values = {}
            for claim in claim_list:
                value_str = str(claim["value"])
                if value_str not in unique_values:
                    unique_values[value_str] = []
                unique_values[value_str].append(claim)
            
            if len(unique_values) == 1:
                # 🔥 EPISTEMIC: Preserve all evidence variants (consensus)
                value_str = list(unique_values.keys())[0]
                variants = unique_values[value_str]
                
                resolved[key] = {
                    "value": variants[0]["value"],
                    "variants": [
                        {
                            "evidence": v["evidence"],
                            "confidence": v["confidence"],
                            "branch_id": v["branch_id"]
                        }
                        for v in variants
                    ]
                }
                logger.debug(f"✅ [Merge] Consensus on '{key}': {resolved[key]['value']} ({len(variants)} variants)")
            else:
                # Conflict: forks disagree
                conflicts[key] = []
                for value_str, variants in unique_values.items():
                    conflicts[key].extend(variants)
                logger.warning(f"⚠️ [Merge] Conflict on '{key}': {len(conflicts[key])} different values")
        
        # 🔥 EPISTEMIC: Uncertainty reduces coverage, never invalidates claims
        uncertain_coverage = set(u["topic"] for u in all_uncertain)
        final_coverage = list(all_coverage - uncertain_coverage)
        
        # Build MergedContract
        merged_contract = MergedContract(
            resolved=resolved,
            conflicts=conflicts,
            coverage=final_coverage,
            uncertain_topics=[UncertainTopic(**u) for u in all_uncertain],
            total_forks=len(fork_results)
        )
        
        # 🔥 EPISTEMIC METRICS (internal, not exposed to user)
        total_claims = len(resolved) + len(conflicts)
        total_uncertain = len(all_uncertain)
        
        if total_claims + total_uncertain > 0:
            omission_rate = total_uncertain / (total_claims + total_uncertain)
            logger.info(f"📊 [Merge] Epistemic metrics:")
            logger.info(f"   Omission rate: {omission_rate:.2%}")
        
        # Count inferred claims
        inferred_count = sum(
            1 for claim_variants in resolved.values()
            for variant in claim_variants.get("variants", [])
            if any(ev.get("type") == "inferred" for ev in variant.get("evidence", []))
        )
        
        if inferred_count + total_uncertain > 0:
            inference_ratio = inferred_count / (inferred_count + total_uncertain)
            logger.info(f"   Inference ratio: {inference_ratio:.2%} (higher = better reasoning)")
        
        logger.info(f"📊 [Merge] Contract merge summary:")
        logger.info(f"   ✅ Resolved: {len(resolved)} claims")
        logger.info(f"   ⚠️  Conflicts: {len(conflicts)} claims")
        logger.info(f"   📋 Coverage: {len(final_coverage)} topics")
        logger.info(f"   ❓ Uncertain: {total_uncertain} topics")
        
        return merged_contract.model_dump()



class SemanticSynthesisState(AsyncHierarchicalState):
    """
    Optional synthesis state that consolidates fork contracts into natural language.
    
    Only active when parallel execution is enabled.
    Uses pluggable synthesis strategy (LLM, rules, hybrid).
    """
    
    def __init__(self, parent, synthesis_strategy):
        super().__init__(parent)
        self.synthesis_strategy = synthesis_strategy
    
    async def handle(self, context: AsyncExecutionContext):
        logger.info("🧬 [Synthesis] Consolidating fork outputs...")
        
        # Get merged contracts
        research_context = await context.get_memory("research_context")
        
        if not research_context or not isinstance(research_context, dict):
            logger.warning("⚠️ [Synthesis] No research context to synthesize")
            return Transition(to="RouterState", reason="No synthesis needed")
        
        # Build synthesis request
        from finitestatemachineAgent.synthesis_contracts import SynthesisRequest
        
        # Extract text from contracts
        fork_outputs = []
        
        # Add resolved claims
        if research_context.get("resolved"):
            resolved_text = "**Consensus Findings:**\n"
            for key, claim in research_context["resolved"].items():
                resolved_text += f"- {key}: {claim['value']}\n"
                if claim.get('evidence'):
                    resolved_text += f"  Evidence: {', '.join(claim['evidence'])}\n"
            fork_outputs.append(resolved_text)
        
        # Add conflicts
        if research_context.get("conflicts"):
            conflicts_text = "**Conflicting Findings:**\n"
            for key, claims in research_context["conflicts"].items():
                conflicts_text += f"- {key}:\n"
                for i, claim in enumerate(claims, 1):
                    conflicts_text += f"  {i}. {claim['value']} (from {claim.get('branch_id', 'unknown')})\n"
            fork_outputs.append(conflicts_text)
        
        # Add coverage and omissions
        if research_context.get("coverage"):
            fork_outputs.append(f"**Topics Covered:** {', '.join(research_context['coverage'])}")
        
        if research_context.get("omissions"):
            fork_outputs.append(f"**Information Gaps:** {', '.join(research_context['omissions'])}")
        
        if not fork_outputs:
            logger.warning("⚠️ [Synthesis] No outputs to synthesize")
            return Transition(to="RouterState", reason="Empty research context")
        
        request = SynthesisRequest(
            task_description=context.user_query,
            fork_outputs=fork_outputs,
            constraints=[],
            output_format="markdown"
        )
        
        # Synthesize using strategy
        try:
            result = await self.synthesis_strategy.synthesize(request)
            
            # 🔥 Structural Guardrails (NOT semantic)
            if not result.answer or len(result.answer.strip()) == 0:
                raise ValueError("Empty synthesis result")
            
            if len(result.answer) > 50000:  # Token limit approximation
                logger.warning(f"⚠️ [Synthesis] Result too long ({len(result.answer)} chars), truncating")
                result.answer = result.answer[:50000] + "\n\n[Response truncated due to length]"
            
            # Store synthesis result
            await context.set_memory("synthesis_result", {
                "answer": result.answer,
                "confidence": result.confidence,
                "gaps": result.gaps,
                "inconsistencies": result.inconsistencies
            })
            
            logger.info(f"✅ [Synthesis] Consolidated {len(fork_outputs)} outputs into {len(result.answer)} chars")
            
            return Transition(to="RouterState", reason="Synthesis complete")
            
        except Exception as e:
            logger.error(f"❌ [Synthesis] Failed: {e}")
            # Fallback: skip synthesis, use raw contracts
            logger.info("🔄 [Synthesis] Falling back to raw contracts")
            return Transition(to="RouterState", reason="Synthesis failed, using raw contracts")


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

        # 1. Inject Context (Synthesis FIRST, then Contracts)
        synthesis_result = await context.get_memory("synthesis_result")
        
        if synthesis_result and synthesis_result.get("answer"):
            # Inject synthesized context
            logger.info(f"📚 [Answer] Injecting synthesized context ({len(synthesis_result['answer'])} chars)")
            logger.info(f"🔍 [Answer] Synthesis preview: {synthesis_result['answer'][:100]}...")
            messages.append({
                "role": "system",
                "content": f"""Use the following synthesized research findings to answer the user's question.

CRITICAL INSTRUCTIONS:
1. **LANGUAGE CONSISTENCY**: You MUST answer in the SAME LANGUAGE as the user's query.
   - If user asks in Portuguese -> Answer in Portuguese
   - If user asks in English -> Answer in English
   - If user asks in Spanish -> Answer in Spanish
   - DO NOT translate technical terms if they are standard in the field, but explain them if needed.

2. **Use the findings**: The provided text is a synthesis of multiple sources. Use it as the primary source of truth.
3. **Natural presentation**: Present the answer naturally to the user.

SYNTHESIZED FINDINGS:
{synthesis_result['answer']}"""
            })
            
        else:
            # Fallback to research context (contracts)
            research_context = await context.get_memory("research_context")
            if research_context and isinstance(research_context, dict):
                # Check if it's a MergedContract (has 'resolved' and 'conflicts' keys)
                if "resolved" in research_context and "conflicts" in research_context:
                    logger.info(f"📚 [Answer] Injecting contract from {research_context.get('total_forks', 0)} fork(s)")
                    
                    # Format contract for LLM
                    contract_text = "# Research Contracts\n\n"
                    
                    # Resolved claims (consensus)
                    if research_context.get("resolved"):
                        contract_text += "## ✅ RESOLVED CLAIMS (Consensus)\n\n"
                        for key, claim in research_context["resolved"].items():
                            contract_text += f"- **{key}**: {claim['value']}\n"
                            contract_text += f"  - Evidence: {', '.join(claim.get('evidence', []))}\n"
                            contract_text += f"  - Confidence: {claim.get('confidence', 1.0):.2f}\n\n"
                    
                    # Conflicts (disagreements)
                    if research_context.get("conflicts"):
                        contract_text += "## ⚠️ CONFLICTS (Multiple Values Found)\n\n"
                        for key, claims in research_context["conflicts"].items():
                            contract_text += f"- **{key}**:\n"
                            for i, claim in enumerate(claims, 1):
                                contract_text += f"  {i}. {claim['value']} (from {claim.get('branch_id', 'unknown')})\n"
                                contract_text += f"     - Evidence: {', '.join(claim.get('evidence', []))}\n"
                            contract_text += "\n"
                    
                    # Coverage and omissions
                    if research_context.get("coverage"):
                        contract_text += f"## 📋 Coverage\n{', '.join(research_context['coverage'])}\n\n"
                    
                    if research_context.get("omissions"):
                        contract_text += f"## ❌ Omissions\n{', '.join(research_context['omissions'])}\n\n"
                    
                    # 🔥 DEBUG: Log contract being injected
                    logger.info(f"📝 [Answer] Contract text being injected ({len(contract_text)} chars):")
                    
                    messages.append({
                        "role": "system",
                        "content": """You have access to research contracts from parallel investigations. Use this information to answer the user's question.

CRITICAL INSTRUCTIONS:
1. **LANGUAGE CONSISTENCY**: You MUST answer in the SAME LANGUAGE as the user's query.
   - If user asks in Portuguese -> Answer in Portuguese
   - If user asks in English -> Answer in English
   - If user asks in Spanish -> Answer in Spanish
   - DO NOT translate technical terms if they are standard in the field, but explain them if needed.

2. **Use the information**: Leverage RESOLVED claims confidently (all sources agreed)
3. **Present conflicts naturally**: If there are CONFLICTS, present all viewpoints without choosing
4. **Be transparent about gaps**: Mention OMISSIONS clearly when information is missing
5. **Natural presentation**: DO NOT mention "contracts", "claims", "resolved", or "conflicts" in your response
6. **User-facing language**: Present findings as if you researched them directly

Remember: The user should NOT see the internal contract structure. Present information naturally in the CORRECT LANGUAGE."""
                    })
                    messages.append({
                        "role": "user",
                        "content": contract_text
                    })
                else:
                    # Legacy format (old semantic merge)
                    logger.info(f"📚 [Answer] Injecting legacy research from {research_context.get('total_branches', 0)} branches")
                    messages.append({
                        "role": "system",
                        "content": "Use the following research results to answer the user's question:"
                    })
                    messages.append({
                        "role": "user",
                        "content": json.dumps(research_context, ensure_ascii=False, indent=2)
                    })

        # 2. Add User Query
        messages.append({"role": "user", "content": context.user_query})
        
        # 3. Call LLM (Streaming or Non-Streaming)
        enable_streaming = await context.get_memory("enable_streaming", True)
        
        if enable_streaming:
            # Stream response
            logger.info("🌊 [Answer] Streaming initialized")
            # 🛡️ Safety Check Before LLM Call (Streaming)
            await context.increment_llm_call()
            stream = self.llm.chat_stream(messages, context)
            
            # Store stream for consumption
            await context.set_memory("answer_stream", stream)
        else:
            # Non-streaming response
            logger.info("📝 [Answer] Generating complete response")
            # 🛡️ Safety Check Before LLM Call (Non-Streaming)
            await context.increment_llm_call()
            # 🔥 Fix: chat() only takes messages, not context
            response = await self.llm.chat(messages)
            
            # Manually track usage if available
            if "usage" in response:
                await context.accumulate_usage(response["usage"])
                
            final_answer = response.get("content", "")
            
            # Store for access
            await context.set_memory("final_answer", final_answer)
            
            # Create fake stream for compatibility
            async def stream_complete():
                for char in final_answer:
                    yield char
            
            await context.set_memory("answer_stream", stream_complete())
            
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
        max_global_requests: int = 50,
        post_router_hook=None,  # 🔥 NEW: Optional hook to intercept router transitions
        initial_state: Optional[str] = None,  # 🔥 NEW: Custom initial state
        # 🔥 Intent Analysis Config
        enable_intent_analysis: bool = False,  # 🔥 NEW: Enable built-in intent analysis
        intent_analysis_llm: Optional['AsyncLLMClient'] = None  # 🔥 NEW: LLM for intent analysis
    ):
        """
        Initialize Async Agent Engine with HFSM architecture.
        
        Args:
            llm: Async LLM client
            registry: Tool registry
            executor: Async tool executor
            system_instruction: System prompt
            tool_choice: Tool selection mode
            skip_validation: Skip validation state
            validation_fn: Custom validation function
            enable_parallel_planning: Enable parallel execution
            planning_system_prompt: Custom planning prompt
            post_router_hook: Hook after router decisions
            merge_fn: Custom merge function for parallel results
            max_parallel_branches: Max parallel branches per fork
            max_global_requests: Global safety limit for LLM calls
            initial_state: Custom initial state name (default: RouterState)
            enable_intent_analysis: Enable built-in intent analysis before RouterState
            intent_analysis_llm: LLM client for intent analysis (defaults to main llm)
        """
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
        self.max_global_requests = max_global_requests
        
        # 🔥 Custom initial state
        self.initial_state_name = initial_state
        
        # 🔥 Intent Analysis Config
        self.enable_intent_analysis = enable_intent_analysis
        self.intent_analysis_llm = intent_analysis_llm or llm
        
        # 🔥 Safety Config
        self.post_router_hook = post_router_hook  # 🔥 Store hook
        
        # 🔥 Custom state registration system
        self.custom_states = {}  # name -> state instance
        self.transition_overrides = []  # List of override rules

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
            enable_parallel=self.enable_parallel_planning,
            post_hook=self.post_router_hook  # 🔥 Pass hook
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
            
            # 🔥 UPDATED: Use ForkContractState (not ForkSummaryState)
            from finitestatemachineAgent.fork_states import ForkContractState
            self.fork_contract_state = ForkContractState(self.terminal, self.llm)
            self.states["ForkContractState"] = self.fork_contract_state
            # Backward compatibility alias
            self.states["ForkSummaryState"] = self.fork_contract_state
            
            # 🔥 NEW: Add semantic synthesis state
            from finitestatemachineAgent.llm_synthesis_strategy import LLMSynthesisStrategy
            synthesis_strategy = LLMSynthesisStrategy(self.llm, temperature=0.3)
            self.synthesis_state = SemanticSynthesisState(self.reasoning, synthesis_strategy)
            self.states["SemanticSynthesisState"] = self.synthesis_state
            
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
    
    # 🔥 Custom State Registration API
    
    def register_state(self, name: str, state):
        """
        Register a custom state in the engine.
        
        Allows domain-specific agents to inject custom states without modifying engine.
        
        Args:
            name: State name (e.g., "TodoListState")
            state: State instance (must inherit from AsyncHierarchicalState)
        
        Example:
            todo_state = TodoListState(parent=engine.reasoning, llm=llm)
            engine.register_state("TodoListState", todo_state)
        """
        self.custom_states[name] = state
        self.states[name] = state
        logger.info(f"🔧 [Engine] Registered custom state: {name}")
    
    def override_transition(
        self, 
        from_state: str, 
        condition: callable,  # (context, transition) -> bool
        new_target: str
    ):
        """
        Override a transition based on condition.
        
        Allows domain-specific agents to redirect transitions without modifying engine.
        
        Args:
            from_state: Source state name (e.g., "RouterState")
            condition: Function (context, transition) -> bool to check if override applies
            new_target: Target state name to redirect to (e.g., "TodoListState")
        
        Example:
            # Inject TodoList before ParallelPlanning
            engine.override_transition(
                from_state="RouterState",
                condition=lambda ctx, trans: trans.to == "ParallelPlanningState",
                new_target="TodoListState"
            )
        """
        # Validate target state exists
        if new_target not in self.states and new_target not in self.custom_states:
            logger.warning(f"⚠️ [Engine] Override target '{new_target}' not registered yet")
        
        self.transition_overrides.append({
            "from": from_state,
            "condition": condition,
            "target": new_target
        })
        logger.info(f"🔧 [Engine] Override registered: {from_state} -> {new_target}")
    
    async def _analyze_intent(self, context: AsyncExecutionContext):
        """
        Built-in intent analysis (runs before RouterState in main flow).
        
        Analyzes user query + chat history to:
        - Extract user intent
        - Generate structured todo list
        - Detect language
        - Enhance query with context from history
        """
        logger.info("=" * 80)
        logger.info("🧠 [IntentAnalysis] Analyzing user intent...")
        logger.info(f"📝 [IntentAnalysis] Query: {context.user_query[:100]}...")
        logger.info("=" * 80)
        
        try:
            # Get chat history and system instruction
            chat_history = await context.get_memory("chat_history", [])
            system_instruction = await context.get_memory("system_instruction", "")
            current_query = context.user_query
            
            logger.info(f"[IntentAnalysis] Chat history length: {len(chat_history)}")
            
            # Build analysis prompt
            messages = [{
                "role": "system",
                "content": f"""{system_instruction}

TAREFA: Análise de Intenção e Planejamento

Analise o histórico RECENTE e a pergunta atual para criar um plano de execução.
Seja direto e objetivo. Ignore conversas passadas que não sejam relevantes para a tarefa ATUAL.

OBJETIVOS:
1. Identificar a intenção principal do usuário
2. Extrair APENAS o contexto necessário do histórico para resolver a tarefa ATUAL
3. Criar uma Lista de Tarefas (Todo List) técnica e sequencial para guiar a execução

Formato JSON OBRIGATÓRIO:
{{
    "intent": "Resumo conciso do objetivo do usuário",
    "required_context": ["Fato relevante 1 extraído do histórico", "Fato relevante 2"],
    "enhanced_query": "Pergunta do usuário enriquecida com o contexto extraído (se necessário)",
    "todo_list": [
        "Ação 1: O que pesquisar/fazer",
        "Ação 2: O que analisar",
        "Ação 3: Como responder"
    ],
    "language": "Código ISO da língua (pt, en, es...)"
}}

IMPORTANTE:
- "required_context" deve estar vazio se o histórico não for relevante para a pergunta ATUAL.
- "enhanced_query" deve ser a própria pergunta se não houver ambiguidade.
- Mantenha o foco TOTAL na resolução da solicitação ATUAL."""
            }]
            
            # Add chat history
            if chat_history:
                messages.extend(chat_history[-5:])  # Last 5 messages for context
            
            # Add current query
            messages.append({
                "role": "user",
                "content": f"Pergunta atual: {current_query}\n\nRetorne APENAS o JSON da análise, sem texto adicional."
            })
            
            logger.info(f"[IntentAnalysis] Calling LLM...")
            
            # Call LLM for intent analysis
            response = await self.intent_analysis_llm.chat(messages)
            
            # Extract JSON from response (handle markdown code blocks)
            content = response.get("content", "").strip()
            
            logger.info(f"[IntentAnalysis] Raw LLM response (first 200 chars): {content[:200]}...")
            
            # Remove markdown code blocks if present
            if content.startswith("```"):
                lines = content.split("\n")
                content = "\n".join(lines[1:-1])  # Remove first and last line
                logger.info(f"[IntentAnalysis] Removed markdown wrapper")
            
            import json
            analysis = json.loads(content)
            
            # Store analysis in context
            await context.set_memory("intent_analysis", analysis)
            await context.set_memory("todo_list", analysis.get("todo_list", []))
            await context.set_memory("user_language", analysis.get("language", "pt"))
            await context.set_memory("intent_analyzed", True)
            
            # Enhance the query with context
            enhanced_query = analysis.get("enhanced_query", current_query)
            context.user_query = enhanced_query
            
            # 🔥 LOG RESULTS
            logger.info(f"✅ [IntentAnalysis] Intent: {analysis.get('intent', 'unknown')}")
            logger.info(f"🌍 [IntentAnalysis] Language: {analysis.get('language', 'unknown')}")
            logger.info(f"📝 [IntentAnalysis] Todo list ({len(analysis.get('todo_list', []))} items):")
            for i, task in enumerate(analysis.get('todo_list', []), 1):
                logger.info(f"   {i}. {task}")
            
            # Log context from history if available
            # 🔥 Handle new key 'required_context' or old 'context_from_history'
            context_items = analysis.get('required_context') or analysis.get('context_from_history', [])
            
            # Store normalized context
            analysis['context_from_history'] = context_items
            await context.set_memory("intent_analysis", analysis)
            
            if context_items:
                logger.info(f"📚 [IntentAnalysis] Context from history:")
                for item in context_items:
                    logger.info(f"   - {item}")
            
            logger.info(f"🔄 [IntentAnalysis] Enhanced query: {enhanced_query[:100]}...")
            logger.info("=" * 80)
            
        except Exception as e:
            logger.error(f"❌ [IntentAnalysis] Failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            # Set flag even on error to prevent infinite loop
            await context.set_memory("intent_analyzed", True)
            await context.set_memory("user_language", "pt")  # Default

    async def dispatch(self, context: AsyncExecutionContext):
        """
        Async state machine dispatch loop with transition resolution.
        """
        # Check if in fork
        is_fork = await context.get_memory("branch_id") is not None
        
        # 🔥 Run intent analysis ONLY in main flow (not in forks)
        if self.enable_intent_analysis and not is_fork:
            intent_analyzed = await context.get_memory("intent_analyzed")
            if not intent_analyzed:
                logger.info("🎯 [Engine] Running built-in intent analysis...")
                await self._analyze_intent(context)
        
        # 🔥 Start from custom initial state ONLY if not in a fork
        # Forks always start from RouterState
        if self.initial_state_name and self.initial_state_name in self.states and not is_fork:
            current_state = self.states[self.initial_state_name]
            logger.info(f"🎯 [Engine] Starting from custom initial state: {self.initial_state_name}")
        else:
            current_state = self.router_state

        while current_state:
            state_name = type(current_state).__name__
            logger.info(f"📍 [Engine] Current state: {state_name}")

            # Handle state (async)
            result = await current_state.handle(context)

            if result is None:
                logger.info(f"🏁 [Engine] Reached terminal state: {state_name}")
                break
            
            # 🔥 Apply transition overrides (custom state injection)
            if isinstance(result, Transition):
                original_target = result.to
                
                for override in self.transition_overrides:
                    if override["from"] == state_name:
                        try:
                            if override["condition"](context, result):
                                result = Transition(
                                    to=override["target"],
                                    reason=f"Override: {result.reason}",
                                    metadata=result.metadata
                                )
                                logger.info(f"🔀 [Override] {state_name}: {original_target} -> {result.to}")
                                break
                        except Exception as e:
                            logger.error(f"❌ [Override] Condition failed: {e}")
                            # Continue without override

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
