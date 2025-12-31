from __future__ import annotations

import json
import logging
import time
import os
from datetime import datetime
from abc import ABC, abstractmethod
from typing import Generator, Optional, List, Dict, Any, Union
from concurrent.futures import ThreadPoolExecutor, as_completed

from core.context import ExecutionContext

# Setup logging
# Setup logging
logger = logging.getLogger("AgentEngine")
logger.setLevel(logging.INFO)
if not logger.handlers:
    # Stream Handler (Console)
    sh = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    
    # File Handler (logs/agent.log)
    if not os.path.exists("logs"):
        os.makedirs("logs", exist_ok=True)
    fh = logging.FileHandler("logs/agent.log", encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

# 🔹 Phase 9: Transition Map
ALLOWED_TRANSITIONS = {
    "Start": ["RouterState"],
    "RouterState": ["ToolState", "AnswerState"],
    "ToolState": ["ValidationState"],
    "ValidationState": ["RetryState", "AnswerState"],
    "RetryState": ["RouterState", "FailState"],
    "ContextPolicyState": ["ReasoningState", "ExecutionState", "TerminalState", "RecoveryState"], # Parent/Middle
    "AgentRootState": ["FailState"], # Parent/Middle
}

# =================================================================================================
# 🔹 Phase 1: Hierarchy Fundamentals
# =================================================================================================

class HierarchicalState(ABC):
    """
    Base class for all states in the Hierarchical Finite State Machine.
    Supports parent-child relationships and delegation.
    """
    def __init__(self, parent: Optional[HierarchicalState] = None):
        self.parent = parent

    @abstractmethod
    def handle(self, context: ExecutionContext) -> Optional[HierarchicalState]:
        """
        Process the current context and return the next state.
        If None is returned, the event is delegated to the parent.
        """
        pass

    def on_enter(self, context: ExecutionContext):
        """Optional hook called when entering this state."""
        pass

    def on_exit(self, context: ExecutionContext):
        """Optional hook called when exiting this state."""
        pass

    def find_state_by_type(self, type_name: str) -> HierarchicalState:
        """Traverse up the hierarchy to find a state provider."""
        if self.parent:
            return self.parent.find_state_by_type(type_name)
        raise Exception(f"State provider for {type_name} not found in hierarchy.")

# =================================================================================================
# 🔹 Phase 1.5: Context Policy & Pruning
# =================================================================================================

class ContextPruner:
    """
    Logic for pruning execution context to prevent window overflows.
    """
    def __init__(self, strategy: str = "cut_last_n"):
        self.strategy = strategy

    def prune(self, context: ExecutionContext):
        """
        Modifies context memory to include a 'pruned_history' view or modifies in place.
        For safety, we will create a 'pruned_tool_calls' memory entry that Router can use,
        preserving the original 'tool_calls' for the Answer state.
        """
        if not context.tool_calls:
            return

        # Strategy: Keep full content only for the last 4 interactions. Truncate others.
        if self.strategy == "cut_last_n":
            total_calls = len(context.tool_calls)
            pruned_calls = []
            
            for i, call in enumerate(context.tool_calls):
                # Check if this is a "recent" call (e.g., last 4)
                is_recent = (total_calls - i) <= 4
                
                # Create a shallow copy to modify result for display/prompt only
                call_copy = call.copy()
                
                raw_result = str(call.get("result", ""))
                if not is_recent and len(raw_result) > 200:
                    call_copy["result"] = raw_result[:200] + "... [TRUNCATED - OLD CONTEXT]"
                
                pruned_calls.append(call_copy)
            
            context.set_memory("active_tool_calls", pruned_calls)

class ContextPolicyState(HierarchicalState):
    """
    Middleware state that enforces context policies (pruning) before delegation.
    Should be the parent of Reasoning, Execution, etc.
    """
    def __init__(self, parent: Optional[HierarchicalState] = None, strategy: str = "cut_last_n"):
        super().__init__(parent)
        self.pruner = ContextPruner(strategy)

    def on_enter(self, context: ExecutionContext):
        # Trigger pruning
        # print("✂️ [Policy] Enforcing context limits...")
        self.pruner.prune(context)
        active_calls = context.get_memory("active_tool_calls", [])
        if active_calls:
            logger.info(f"✂️ [Policy] Pruned context. Active tool calls: {len(active_calls)}")

    def handle(self, context: ExecutionContext) -> Optional[HierarchicalState]:
        # Just delegate to active child (which is handled by dispatch loop)
        return None

class AgentRootState(HierarchicalState):
    """
    The root of the hierarchy. Handles global issues (fatal errors, generic fallbacks).
    Doesn't have a parent.
    """
    def __init__(self):
        super().__init__(parent=None)

    def handle(self, context: ExecutionContext) -> Optional[HierarchicalState]:
        # If we reached the root without a transition, it's a deadlock or failure.
        logger.error("❌ [Root] No state handled the context. Terminating.")
        return FailState(self)
        
class ReasoningState(HierarchicalState):
    """Parent for logic that involves thinking/routing."""
    def handle(self, context: ExecutionContext) -> Optional[HierarchicalState]:
        return None # Delegate to parent

class ExecutionState(HierarchicalState):
    """Parent for tool execution and validation."""
    def handle(self, context: ExecutionContext) -> Optional[HierarchicalState]:
        return None # Delegate to parent

class RecoveryState(HierarchicalState):
    """Parent for handling retries and errors."""
    def handle(self, context: ExecutionContext) -> Optional[HierarchicalState]:
        return None # Delegate to parent

class TerminalState(HierarchicalState):
    """Marker for final states (Answer/Fail). Stops the engine."""
    def handle(self, context: ExecutionContext) -> Optional[HierarchicalState]:
        return None # Delegate to parent

# =================================================================================================
# 🔹 Phase 4: Migrate Substates
# =================================================================================================

class RouterState(ReasoningState):
    def __init__(self, parent: HierarchicalState, llm, registry):
        super().__init__(parent)
        self.llm = llm
        self.registry = registry

    def handle(self, context: ExecutionContext) -> Optional[HierarchicalState]:
        system_instruction = context.get_memory("system_instruction", "")
        
        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": context.user_query},
        ]

        # Reconstruct chat history if available (not implemented generically in context yet)
        if hasattr(context, "chat_history") and context.chat_history:
             pass

        # Reconstruct history with tool calls (PRUNED FOR ROUTER via ContextPolicy)
        # We look for "active_tool_calls" which is populated by ContextPolicyState
        tool_calls_to_use = context.get_memory("active_tool_calls", context.tool_calls)
        
        if not tool_calls_to_use:
            logger.warning("⚠️ [Router] No active tool calls found in memory.")

        for call in tool_calls_to_use:
            tool_call_id = f"call_{call['tool_name']}_{call.get('iteration', 0)}"
            # Handle patched IDs from pruner if needed, but here we consistently rebuild
            
            # Log reconstruction details
            logger.debug(f"🧩 [Router] Reconstructing tool call: {call['tool_name']}")
            
            messages.append({
                "role": "assistant",
                "content": None, 
                "tool_calls": [
                    {
                        "id": tool_call_id,
                        "type": "function",
                        "function": {
                            "name": call["tool_name"],
                            "arguments": json.dumps(call["arguments"], ensure_ascii=False)
                        }
                    }
                ]
            })

            content_str = str(call.get("result", ""))
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call_id,
                "name": call["tool_name"],
                "content": content_str
            })

        logger.info("🧠 [Router] Thinking...")
        response = self.llm.chat_with_tools(
            messages=messages,
            tools=self.registry.to_openai_format()
        )
        
        # Accumulate usage
        usage = response.get("usage", {})
        logger.info(f"📊 [Router] Token usage: {usage}")
        
        total_usage = context.get_memory("total_usage", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
        total_usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
        total_usage["completion_tokens"] += usage.get("completion_tokens", 0)
        total_usage["total_tokens"] += usage.get("total_tokens", 0)
        context.set_memory("total_usage", total_usage)

        if response.get("tool_calls"):
            context.set_memory("pending_tool_calls", response["tool_calls"])
            # Transition to ToolState (which is under ExecutionState)
            return ToolState(self.parent.find_state_by_type("ExecutionState"), context.get_memory("executor"))
        
        logger.warning("[Router] No tool calls generated by LLM.")
        context.set_memory("last_llm_content", response.get("content", ""))
        return AnswerState(self.parent.find_state_by_type("TerminalState"), self.llm)

class ToolState(ExecutionState):
    def __init__(self, parent: HierarchicalState, executor, max_workers: int = 4):
        super().__init__(parent)
        self.executor = executor
        self.pool = ThreadPoolExecutor(max_workers=max_workers)

    def handle(self, context: ExecutionContext) -> Optional[HierarchicalState]:
        futures_map = {}
        pending_calls = context.get_memory("pending_tool_calls", [])

        if not pending_calls:
             return None 

        for call in pending_calls:
            name = call["function"]["name"]
            try:
                args = json.loads(call["function"]["arguments"])
            except:
                args = {}
            
            logger.info(f"🛠️ [Tool] Executing: {name} with args {args}")
            future = self.pool.submit(self.executor.execute, name, args)
            futures_map[future] = call

        for future in as_completed(futures_map):
            call = futures_map[future]
            try:
                result_map = future.result()
                result = result_map.get("result") if result_map.get("success") else result_map.get("error")
                logger.info(f"✅ [Tool] Result for {call['function']['name']}: {str(result)[:100]}...")
            except Exception as e:
                result = str(e)
                logger.error(f"❌ [Tool] Error executing {call['function']['name']}: {str(e)}")
                
            name = call["function"]["name"]
            try:
                args = json.loads(call["function"]["arguments"])
            except:
                args = {}
                
            context.add_tool_call(name, args, result)

        context.set_memory("pending_tool_calls", [])
        
        return ValidationState(self.parent, context.get_memory("llm"))


class ValidationState(ExecutionState):
    def __init__(self, parent: HierarchicalState, llm):
        super().__init__(parent)
        self.llm = llm

    def handle(self, context: ExecutionContext) -> Optional[HierarchicalState]:
        logger.info("🔍 [Validation] Checking data...")
        prompt = f"""
        Você é um validador lógico.
        Verifique se as informações coletadas são suficientes para responder.
        
        PERGUNTA:
        {context.user_query}
        
        DADOS:
        {json.dumps(context.tool_calls, ensure_ascii=False, default=str)}
        
        Responda apenas:
        {{"valid": true}} ou {{"valid": false}}
        """
        logger.debug(f"📝 [Validation] Prompt: {prompt}")

        try:
            response_dict = self.llm.chat([{"role": "user", "content": prompt}])
            
            # Accumulate usage
            usage = response_dict.get("usage", {})
            total_usage = context.get_memory("total_usage", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
            total_usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
            total_usage["completion_tokens"] += usage.get("completion_tokens", 0)
            total_usage["total_tokens"] += usage.get("total_tokens", 0)
            context.set_memory("total_usage", total_usage)
            
            response_content = response_dict.get("content", "")
            
            clean_resp = response_content.replace("```json", "").replace("```", "").strip()
            result = json.loads(clean_resp)
            is_valid = result.get("valid", False)
            logger.info(f"✅ [Validation] Result: {is_valid}")
        except Exception as e:
            logger.error(f"❌ [Validation] Failed to parse LLM response: {e}")
            is_valid = False

        if is_valid:
             # Go to Answer
             return AnswerState(self.parent.find_state_by_type("TerminalState"), self.llm)
        else:
             # Go to Retry
             return RetryState(self.parent.find_state_by_type("RecoveryState"))


class RetryState(RecoveryState):
    def __init__(self, parent: HierarchicalState):
        super().__init__(parent)

    def handle(self, context: ExecutionContext) -> Optional[HierarchicalState]:
        logger.warning(f"⚠️ [Retry] Attempting recovery...")
        retry_count = context.get_memory("retry_count", 0)
        
        # Configurable Retries (Phase 12)
        # Check specific config -> global config -> default
        config = context.get_memory("state_config", {}).get("RetryState", {})
        max_retries = config.get("max_retries", context.get_memory("max_retries", 2))
        
        retry_count += 1
        logger.warning(f"⚠️ [Retry] Attempt {retry_count}/{max_retries}")
        context.set_memory("retry_count", retry_count)
        
        if retry_count > max_retries:
            logger.error("❌ [Retry] Max retries reached. Transitioning to FailState.")
            return FailState(self.parent.find_state_by_type("TerminalState"))

        context.user_query = f"Refine melhor:\n{context.user_query}"
        
        # Back to Router (in Reasoning)
        return RouterState(self.parent.find_state_by_type("ReasoningState"), context.get_memory("llm"), context.get_memory("registry"))


class AnswerState(TerminalState):
    def __init__(self, parent: HierarchicalState, llm):
        super().__init__(parent)
        self.llm = llm
        self.generator = None

    def handle(self, context: ExecutionContext) -> Optional[HierarchicalState]:
        """
        Constructs the final prompt with full context and initializes the streaming generator.
        
        This state does NOT return a next state immediately. It prepares the generator
        Which the AgentEngine will yield from.
        """
        logger.info("✅ [Answer] Generating final response...")
        logger.debug(f"📝 [Answer] Tool calls in context: {len(context.tool_calls)}")
        
        system_instruction = context.get_memory("system_instruction", "")
        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": context.user_query},
        ]
        
        # Append tool interactions to context
        for call in context.tool_calls:
            tool_call_id = f"call_{call['tool_name']}_{call.get('iteration', 0)}"
            messages.append({
                "role": "assistant",
                "content": None, 
                "tool_calls": [
                    {"id": tool_call_id, "type": "function", "function": {"name": call["tool_name"], "arguments": json.dumps(call["arguments"])}}
                ]
            })
            content_str = str(call.get("result", ""))
            messages.append({"role": "tool", "tool_call_id": tool_call_id, "name": call["tool_name"], "content": content_str})

        messages.append({
            "role": "system",
            "content": "Based on the tool results above, provide a clear and direct answer to the user's question. Do NOT call any more tools. Just answer."
        })

        # Generator that streams and updates context usage side-effect
        logger.info("🌊 [Answer] Streaming initialized")
        self.generator = self.llm.chat_stream(messages, context=context)
        return None # Stay here / Finish

class FailState(TerminalState):
    def __init__(self, parent: HierarchicalState):
        super().__init__(parent)
        self.generator = None

    def handle(self, context: ExecutionContext) -> Optional[HierarchicalState]:
        logger.error("❌ [Fail] Terminating agent.")
        def fail_gen():
            yield "Erro: Não foi possível obter as informações necessárias após várias tentativas."
        self.generator = fail_gen()
        return None 

# =================================================================================================
# 🔹 Phase 2 & 8: Generic Engine
# =================================================================================================

class AgentEngine:
    def __init__(
        self,
        llm,
        registry,
        executor,
        system_instruction: str = ""
    ):
        self.llm = llm
        self.registry = registry
        self.executor = executor
        self.system_instruction = system_instruction
        
        self.states = {}
        
        # Initialize Hierarchy Root
        self.root = AgentRootState()
        self.states["AgentRootState"] = self.root
        
        # Policy State (Middleware)
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
        
        # Allow states to verify/find peers (basic service locator via root)
        self.root.find_state_by_type = self._find_state_provider

    def register_state(self, name: str, state_instance: HierarchicalState):
        """Dynamically register a state."""
        self.states[name] = state_instance

    def _find_state_provider(self, type_name: str) -> HierarchicalState:
        return self.states.get(type_name, self.root)

    def _before_handle(self, state: HierarchicalState, context: ExecutionContext):
        """Hook called before state handling."""
        state_name = state.__class__.__name__
        logger.debug(f"➡️ Entering state: {state_name}")
        if context.metrics.get("state_visits") is None:
            context.metrics["state_visits"] = {}
        context.metrics["state_visits"][state_name] = context.metrics["state_visits"].get(state_name, 0) + 1
        
    def _after_handle(self, state: HierarchicalState, context: ExecutionContext, duration: float):
        """Hook called after state handling."""
        state_name = state.__class__.__name__
        logger.debug(f"⬅️ Exiting state: {state_name} (took {duration:.4f}s)")
        
    def _on_transition(self, from_state: HierarchicalState, to_state: HierarchicalState, context: ExecutionContext):
        """Hook called on state transition."""
        from_name = from_state.__class__.__name__
        to_name = to_state.__class__.__name__
        
        # Validation
        valid_targets = ALLOWED_TRANSITIONS.get(from_name, [])
        # We also allow transitions if 'to_name' is a substate of allowed targets?
        # Simpler check: if mapped, check exact. If not mapped (like abstract/middleware), skipped.
        if from_name in ALLOWED_TRANSITIONS:
             if to_name not in valid_targets and "FailState" not in to_name:
                 logger.warning(f"⚠️ Possible invalid transition: {from_name} -> {to_name}")

        logger.info(f"🔄 Transition: {from_name} -> {to_name}")
        
        # Persistence: Save snapshot
        self._save_snapshot(context, f"transition_{from_name}_to_{to_name}")

    def _save_snapshot(self, context: ExecutionContext, event_name: str):
        """Save context snapshot to disk/log."""
        try:
            snapshot = context.snapshot()
            
            # Phase 10: Disk Persistence
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"snapshot_{timestamp}_{event_name}.json"
            directory = "logs/snapshots"
            
            if not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
                
            filepath = os.path.join(directory, filename)
            
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(snapshot, f, ensure_ascii=False, indent=2)
                
            logger.debug(f"💾 Snapshot saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save snapshot: {e}")

    def dispatch(self, state: HierarchicalState, context: ExecutionContext) -> HierarchicalState:
        """
        The generic dispatch loop. Bubbles events up if handle() returns None.
        Returns the NEXT state.
        Handles exceptions via global try/catch.
        """
        start_state = state
        
        # Hook: Check lineage for ContextPolicyState and trigger on_enter
        temp = state
        while temp:
            if isinstance(temp, ContextPolicyState):
                temp.on_enter(context)
                break
            temp = temp.parent
            
        # Hook: Trigger on_enter for the active state
        start_state.on_enter(context)
        
        while state:
            try:
                # Pre-handle hook
                self._before_handle(state, context)
                start_time = time.time()
                
                next_state = state.handle(context)
                
                # Post-handle hook
                duration = time.time() - start_time
                self._after_handle(state, context, duration)
                
                if next_state:
                    self._on_transition(state, next_state, context)
                    return next_state # Transition found
                
                # Delegate to parent
                if state.parent:
                    # logger.debug(f"⬆️ Delegating from {state.__class__.__name__} to {state.parent.__class__.__name__}")
                    state = state.parent
                else:
                    # Root reached and returned None -> Stop or Error?
                    if isinstance(start_state, TerminalState):
                        return start_state 
                    
                    logger.error(f"❌ Deadlock: {start_state.__class__.__name__} exhausted hierarchy without transition.")
                    return FailState(self.root) # Fallback

            except Exception as e:
                logger.error(f"💥 Exception in {state.__class__.__name__}: {str(e)}", exc_info=True)
                return FailState(self.root)

        return FailState(self.root)

    def resume_from_snapshot(self, filepath: str, new_query: Optional[str] = None) -> tuple[Generator[str, None, None], ExecutionContext]:
        """
        Resume execution from a saved snapshot file.
        """
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            context = ExecutionContext.load_from_snapshot(data)
            logger.info(f"📂 Loaded context from {filepath}")
            
            if new_query:
                context.user_query = new_query
                logger.info(f"🔄 Updating query to: {new_query}")
                
            # Re-inject dependencies (since they are not serializable)
            context.set_memory("llm", self.llm)
            context.set_memory("registry", self.registry)
            context.set_memory("executor", self.executor)
            
            # Determine start state (naive: start at Router for safety)
            # In a real system, we might persist 'current_state' in memory and map back.
            start_state = RouterState(self.reasoning, self.llm, self.registry)
            
            # Start loop
            # This duplicates run_stream logic partially, ideally refactor common loop.
            # For now, inline loop logic or call internal runner? 
            # run_stream initializes context. Here context exists.
            # We can refactor run_stream to accept context.
            return self._run_loop(start_state, context)
            
        except Exception as e:
             logger.error(f"Failed to resume: {e}")
             raise e

    def generate_state_graph(self) -> str:
        """Generate Mermaid graph for the state machine."""
        lines = ["stateDiagram-v2"]
        for src, targets in ALLOWED_TRANSITIONS.items():
            for tgt in targets:
                lines.append(f"    {src} --> {tgt}")
        return "\n".join(lines)

    def _run_loop(self, current_state: HierarchicalState, context: ExecutionContext) -> tuple[Generator[str, None, None], ExecutionContext]:
        """Internal run loop reusable by run_stream and resume."""
        while True:
            # Dispatch returns the NEW state (or the same if terminal)
            next_state = self.dispatch(current_state, context)
            
            # Check if Terminal
            if isinstance(next_state, TerminalState):
                # Ensure generator is ready
                gen = next_state.generator
                if not gen:
                     # Should have been set in handle()
                     next_state.handle(context) 
                     gen = next_state.generator
                
                # CLEANUP: Remove non-serializable objects from memory before returning context to API
                context.memory.pop("llm", None)
                context.memory.pop("registry", None)
                context.memory.pop("executor", None)
                
                return gen, context

            current_state = next_state
    
    def run_stream(
        self, 
        query: str,
        chat_history: Optional[list] = None
    ) -> tuple[Generator[str, None, None], ExecutionContext]:
        
        context = ExecutionContext(user_query=query)
        context.set_memory("system_instruction", self.system_instruction)
        context.set_memory("chat_history", chat_history or [])
        context.set_memory("retry_count", 0)
        context.set_memory("max_retries", 2)
        context.set_memory("pending_tool_calls", [])
        
        # Inject dependencies into memory for states to access when recreating siblings
        context.set_memory("llm", self.llm)
        context.set_memory("registry", self.registry)
        context.set_memory("executor", self.executor)

        # Initial State
        current_state = RouterState(self.reasoning, self.llm, self.registry)
        
        return self._run_loop(current_state, context)
