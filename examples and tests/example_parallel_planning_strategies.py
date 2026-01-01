"""
Example: Custom Parallel Planning for RAG Agent
================================================

Shows different strategies for customizing parallel execution planning.
"""

from finitestatemachineAgent.hfsm_agent_async import ParallelPlan, BranchSpec


# Strategy 1: Pure deterministic logic (no LLM)
async def deterministic_planner(context):
    """
    Deterministic planner based on keywords.
    No LLM calls, pure logic.
    """
    query_lower = context.user_query.lower()
    
    # Check for comparison keywords
    if any(word in query_lower for word in ["compare", "versus", "vs", "difference between"]):
        # Extract topics (simplified)
        if "compare" in query_lower:
            parts = query_lower.split("compare")[1].split("and")
            if len(parts) >= 2:
                return ParallelPlan(
                    strategy="parallel_research",
                    branches=[
                        BranchSpec(id=f"topic_{i}", goal=f"Research {part.strip()}")
                        for i, part in enumerate(parts[:3])  # Max 3 branches
                    ],
                    merge_policy="comparison"
                )
    
    # Check for multiple questions
    if "?" in context.user_query:
        questions = [q.strip() + "?" for q in context.user_query.split("?") if q.strip()]
        if len(questions) > 1:
            return ParallelPlan(
                strategy="parallel_research",
                branches=[
                    BranchSpec(id=f"q_{i}", goal=q)
                    for i, q in enumerate(questions[:3])
                ],
                merge_policy="append"
            )
    
    # Default: single execution
    return ParallelPlan(strategy="single")


# Strategy 2A: Override system prompt completely (string)
custom_planning_prompt = """You are an advanced planning AI for financial queries.

Analyze if the query should use parallel research for better results.

FINANCIAL DOMAIN RULES:
- For stock comparisons: ALWAYS parallelize (one branch per stock)
- For complex economic questions: Break into sub-topics
- For simple lookups: Use single execution

Output JSON:
{"strategy": "single"} OR {"strategy": "parallel_research", "branches": [...]}

Query: {query}"""


# Strategy 2B: Enhance default prompt incrementally (callable)
def enhance_planning_prompt(default_prompt, context):
    """
    Callable that receives default prompt and context.
    Returns enhanced prompt.
    """
    # Check query complexity
    query_length = len(context.user_query.split())
    
    if query_length < 10:
        # Keep default for simple queries
        return default_prompt
    
    # Add domain-specific hints for complex queries
    enhancement = f"""

ADDITIONAL CONTEXT:
- Query complexity: {query_length} words (complex)
- Consider divide-and-conquer strategy
- Each branch should research independent aspects
- Parallel execution recommended for multi-faceted questions"""
    
    return default_prompt + enhancement


# Strategy 3: Hybrid (deterministic + LLM fallback)
async def hybrid_planner(context, default_planner):
    """
    Try deterministic rules first, fall back to LLM if needed.
    """
    # First, try deterministic logic
    deterministic_plan = await deterministic_planner(context)
    
    if deterministic_plan.strategy == "parallel_research":
        # Deterministic logic found a pattern
        return deterministic_plan
    
    # No pattern found, use LLM
    return await default_planner(context)


# Strategy 4: Context-aware planner (uses flags/memory)
async def context_aware_planner(context):
    """
    Uses context memory and flags to decide parallelization.
    """
    # Check if user explicitly requested parallel execution
    force_parallel = await context.get_memory("force_parallel", False)
    
    if force_parallel:
        # User wants parallel, create branches from query
        return ParallelPlan(
            strategy="parallel_research",
            branches=[
                BranchSpec(id="main", goal=context.user_query)
            ]
        )
    
    # Check conversation history
    history = await context.get_memory("chat_history", [])
    if len(history) > 5:
        # Long conversation, might benefit from parallel research
        return ParallelPlan(
            strategy="parallel_research",
            branches=[
                BranchSpec(id="current", goal=context.user_query),
                BranchSpec(id="context", goal="Summarize conversation context")
            ]
        )
    
    return ParallelPlan(strategy="single")


# ============================================================================
# Usage Examples
# ============================================================================

"""
# Example 1: Use deterministic planner (no LLM)
agent = AsyncRAGAgentFSM(
    embedding_manager=em,
    enable_parallel_planning=True,
    parallel_plan_fn=deterministic_planner  # Pure logic, no LLM
)

# Example 2A: Override system prompt completely
agent = AsyncRAGAgentFSM(
    embedding_manager=em,
    enable_parallel_planning=True,
    planning_system_prompt=custom_planning_prompt  # String override
)

# Example 2B: Enhance default prompt incrementally
agent = AsyncRAGAgentFSM(
    embedding_manager=em,
    enable_parallel_planning=True,
    planning_system_prompt=enhance_planning_prompt  # Callable enhancement
)

# Example 3: Hybrid approach (deterministic + LLM fallback)
agent = AsyncRAGAgentFSM(
    embedding_manager=em,
    enable_parallel_planning=True,
    parallel_plan_fn=lambda ctx: hybrid_planner(ctx, None)
)

# Example 4: Use default LLM planner (no customization)
agent = AsyncRAGAgentFSM(
    embedding_manager=em,
    enable_parallel_planning=True
    # No parallel_plan_fn or planning_system_prompt -> uses default
)

# Example 5: Context-aware planner
agent = AsyncRAGAgentFSM(
    embedding_manager=em,
    enable_parallel_planning=True,
    parallel_plan_fn=context_aware_planner
)
"""

