"""
Parallel Execution States for HFSM Agent
=========================================

New states to support parallel tool execution with context forking.
"""


class ParallelPlanningState(AsyncHierarchicalState):
    """
    Async parallel planning state with customizable planning logic.
    Decides whether to execute tools in parallel or sequentially.
    """
    def __init__(self, parent, llm, planner_fn=None):
        super().__init__(parent)
        self.llm = llm
        self.planner_fn = planner_fn  # Custom planner function
    
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
            return self.find_state_by_type("RouterState")
        
        # Store plan in context
        await context.set_memory("parallel_plan", plan)
        
        # Decide next state based on strategy
        if plan.strategy == "parallel_research" and plan.branches:
            logger.info(f"🔀 [ParallelPlan] Parallel execution with {len(plan.branches)} branches")
            return self.find_state_by_type("ForkDispatchState")
        else:
            logger.info("➡️ [ParallelPlan] Single path execution")
            return self.find_state_by_type("RouterState")
    
    async def _default_llm_plan(self, context: AsyncExecutionContext) -> ParallelPlan:
        """Default LLM-based planning logic."""
        planning_prompt = """You are a planning module for a tool execution system.

Analyze the user's request and decide if it should be split into independent parallel research branches.

Rules:
- Branches must be INDEPENDENT (no dependencies between them)
- Each branch should research a different aspect
- Only use parallel if it provides clear value
- Do NOT plan tool execution, only research goals

If parallel is NOT useful, respond with:
{"strategy": "single"}

If parallel IS useful, respond with:
{
  "strategy": "parallel_research",
  "branches": [
    {"id": "branch_1", "goal": "Research X", "constraints": []},
    {"id": "branch_2", "goal": "Research Y", "constraints": []}
  ],
  "merge_policy": "append"
}

User query: {query}

Respond with valid JSON only."""

        messages = [
            {"role": "system", "content": planning_prompt.format(query=context.user_query)}
        ]
        
        try:
            response = await self.llm.chat(messages)
            plan_data = json.loads(response["content"])
            
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
        plan = await context.get_memory("parallel_plan")
        if not plan or not plan.branches:
            logger.warning("⚠️ [ForkDispatch] No valid plan, falling back")
            return self.find_state_by_type("RouterState")
        
        logger.info(f"🔀 [ForkDispatch] Spawning {len(plan.branches)} branches")
        
        # Create forked contexts
        fork_contexts = []
        for branch in plan.branches:
            fork_ctx = context.fork()
            await fork_ctx.set_memory("branch_id", branch.id)
            await fork_ctx.set_memory("branch_goal", branch.goal)
            await fork_ctx.set_memory("branch_constraints", branch.constraints)
            
            # Override user query with branch goal
            fork_ctx.user_query = f"{context.user_query}\n\nBranch goal: {branch.goal}"
            
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
            
            # Store results
            await context.set_memory("fork_results", successful_forks)
            
            if successful_forks:
                return self.find_state_by_type("MergeState")
            else:
                logger.warning("⚠️ [ForkDispatch] All forks failed, falling back")
                return self.find_state_by_type("RouterState")
        
        except Exception as e:
            logger.error(f"❌ [ForkDispatch] Parallel execution failed: {e}")
            return self.find_state_by_type("RouterState")
    
    async def _execute_fork(self, branch_id: str, fork_ctx: AsyncExecutionContext):
        """Execute a single fork through the engine."""
        logger.info(f"🌿 [Fork:{branch_id}] Starting execution")
        
        # Start from RouterState for this fork
        initial_state = self.find_state_by_type("RouterState")
        await self.engine.dispatch(initial_state, fork_ctx)
        
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
        plan = await context.get_memory("parallel_plan")
        
        if not fork_results:
            logger.warning("⚠️ [Merge] No fork results to merge")
            return self.find_state_by_type("RouterState")
        
        logger.info(f"🧬 [Merge] Merging {len(fork_results)} fork results")
        
        # Use custom merge function if provided
        if self.merge_fn:
            try:
                merged = await self.merge_fn(context, fork_results, plan)
                logger.info("✅ [Merge] Custom merge completed")
            except Exception as e:
                logger.error(f"❌ [Merge] Custom merge failed: {e}, using default")
                merged = self._default_append_merge(fork_results)
        else:
            merged = self._default_append_merge(fork_results)
            logger.info("✅ [Merge] Default append merge completed")
        
        # Store merged results
        await context.set_memory("merged_context", merged)
        
        # Continue to RouterState with merged context
        return self.find_state_by_type("RouterState")
    
    def _default_append_merge(self, fork_results: List[AsyncExecutionContext]) -> dict:
        """Default merge strategy: simple append of all results."""
        outputs = []
        
        for fork_ctx in fork_results:
            branch_id = fork_ctx.memory.get("branch_id", "unknown")
            outputs.append({
                "branch_id": branch_id,
                "branch_goal": fork_ctx.memory.get("branch_goal", ""),
                "tool_calls": fork_ctx.tool_calls,
                "memory": fork_ctx.memory,
                "snapshot": fork_ctx.snapshot()
            })
        
        return {
            "strategy": "append",
            "branches": outputs,
            "total_branches": len(outputs)
        }
