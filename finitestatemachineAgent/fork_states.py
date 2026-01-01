"""
Research Fork State
====================

Specialized state for fork execution that bypasses the full Router logic.
Forks execute with minimal context and a focused research goal.
"""

import logging
import json
from abc import ABC, abstractmethod
from typing import Optional

logger = logging.getLogger("AsyncAgentEngine")

# Import base classes to avoid circular dependency
from core.context_async import AsyncExecutionContext


class Transition:
    """Transition object for state changes."""
    def __init__(self, to: str, reason: str = "", metadata: dict = None):
        self.to = to
        self.reason = reason
        self.metadata = metadata or {}


class AsyncHierarchicalState(ABC):
    """Base class for async hierarchical states."""
    def __init__(self, parent: Optional['AsyncHierarchicalState'] = None):
        self.parent = parent
    
    @abstractmethod
    async def handle(self, context: AsyncExecutionContext):
        pass
    
    def find_state_by_type(self, type_name: str):
        """Traverse hierarchy to find state."""
        if self.parent:
            return self.parent.find_state_by_type(type_name)
        raise Exception(f"State provider for {type_name} not found")



class ResearchForkState(AsyncHierarchicalState):
    """
    Entry point for fork execution.
    
    Instead of going through RouterState (which would add unnecessary LLM calls),
    forks start here with a focused research goal.
    
    Flow: ResearchForkState -> ToolState (if tools needed) -> ForkSummaryState
    """
    def __init__(self, parent, llm, registry, tool_choice=None):
        super().__init__(parent)
        self.llm = llm
        self.registry = registry
        self.tool_choice = tool_choice
    
    async def handle(self, context: AsyncExecutionContext):
        # Get branch goal from memory (set by ForkDispatchState)
        branch_goal = await context.get_memory("branch_goal", "")
        branch_id = await context.get_memory("branch_id", "unknown")
        
        if not branch_goal:
            logger.error(f"❌ [ResearchFork:{branch_id}] No branch goal found")
            return Transition(to="ForkSummaryState", reason="Missing branch goal")
        
        # 🔥 Enhanced logging for observability
        logger.info("=" * 70)
        logger.info(f"🔬 [ResearchFork:{branch_id}] STARTING FORK EXECUTION")
        logger.info(f"   📋 Task: {branch_goal}")
        logger.info(f"   🎯 Branch ID: {branch_id}")
        logger.info("=" * 70)
        
        # Build focused research prompt
        system_instruction = """You are a research worker executing a specific research task.

Your job:
1. Analyze the research goal
2. Select appropriate tools to gather information
3. Execute efficiently

Do NOT ask questions. Do NOT engage in conversation. Focus on the task."""
        
        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": f"Research goal: {branch_goal}"}
        ]
        
        # Safety check
        await context.increment_llm_call()
        
        # Call LLM with tools
        response = await self.llm.chat_with_tools(
            messages=messages,
            tools=self.registry.to_openai_format(),
            tool_choice=self.tool_choice
        )
        
        # Track usage
        usage = response.get('usage', {})
        if usage:
            await context.accumulate_usage(usage)
        
        # Check for tool calls
        if response.get("tool_calls"):
            logger.info(f"🔧 [ResearchFork:{branch_id}] {len(response['tool_calls'])} tool(s) selected")
            
            # Store tool calls
            for call in response["tool_calls"]:
                await context.add_tool_call(
                    tool_name=call["function"]["name"],
                    arguments=json.loads(call["function"]["arguments"]),
                    result=None
                )
            
            # Execute tools
            return Transition(to="ToolState", reason="Tools selected for research")
        
        # No tools needed - store direct research notes
        elif response.get("content"):
            logger.info(f"📝 [ResearchFork:{branch_id}] Direct research notes provided")
            await context.set_memory("research_notes", response["content"])
            return Transition(to="ForkSummaryState", reason="Direct research completed")
        
        else:
            logger.warning(f"⚠️ [ResearchFork:{branch_id}] No tools or content generated")
            return Transition(to="ForkSummaryState", reason="Empty research result")


class ForkSummaryState(AsyncHierarchicalState):
    """
    Summarizes fork execution results into a structured format.
    
    Output format:
    {
        "branch_id": "...",
        "goal": "...",
        "summary": "...",
        "sources": [...]
    }
    """
    def __init__(self, parent, llm):
        super().__init__(parent)
        self.llm = llm
    
    async def handle(self, context: AsyncExecutionContext):
        logger.info("📊 [ForkSummary] Consolidating research results...")
        
        branch_id = await context.get_memory("branch_id", "unknown")
        branch_goal = await context.get_memory("branch_goal", "")
        research_notes = await context.get_memory("research_notes", "")
        
        # Collect tool results
        sources = []
        summary_parts = []
        
        if context.tool_calls:
            for call in context.tool_calls:
                sources.append({
                    "tool": call["tool_name"],
                    "arguments": call["arguments"],
                    "result": str(call.get("result", ""))[:500]  # Truncate for brevity
                })
                
                # Add to summary
                if call.get("result"):
                    summary_parts.append(f"- {call['tool_name']}: {str(call['result'])[:200]}")
        
        # If we have direct research notes, include them
        if research_notes:
            summary_parts.append(f"- Research notes: {research_notes[:200]}")
        
        # Generate concise summary
        if summary_parts:
            summary = "\n".join(summary_parts)
        else:
            summary = "No results obtained"
        
        # Store structured result
        final_summary = {
            "branch_id": branch_id,
            "goal": branch_goal,
            "summary": summary,
            "sources": sources
        }
        
        await context.set_memory("final_summary", final_summary)
        
        logger.info(f"✅ [ForkSummary:{branch_id}] Summary created with {len(sources)} source(s)")
        
        # Terminal state for fork
        return None
