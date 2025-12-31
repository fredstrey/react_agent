"""
Async RAG Agent HFSM
====================

Async version of RAG Agent using HFSM architecture.
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from finitestatemachineAgent.hfsm_agent_async import AsyncAgentEngine
from providers.llm_client_async import AsyncLLMClient
from core.executor_async import AsyncToolExecutor
from core.registry import ToolRegistry
import tools.rag_tools as rag_tools
from typing import AsyncIterator


class AsyncRAGAgentFSM:
    """
    Async RAG Agent using Hierarchical Finite State Machine.
    
    Uses async/await for better concurrency and performance.
    """
    
    def __init__(
        self,
        embedding_manager,
        model: str = "google/gemini-2.0-flash-exp:free",
        skip_validation: bool = False
    ):
        # Initialize RAG tools
        rag_tools.initialize_rag_tools(embedding_manager)
        
        # Setup registry
        registry = ToolRegistry()
        
        # Register all RAG tools
        tools_list = [
            rag_tools.search_documents,
            rag_tools.get_stock_price,
            rag_tools.compare_stocks,
            rag_tools.redirect
        ]
        
        for tool_func in tools_list:
            if hasattr(tool_func, '_tool_name'):
                registry.register(
                    name=tool_func._tool_name,
                    description=tool_func._tool_description,
                    function=tool_func,
                    args_model=tool_func._args_model
                )
        
        # Create async executor and LLM
        executor = AsyncToolExecutor(registry)
        llm = AsyncLLMClient(model=model)
        
        # System instruction
        system_instruction = """
Você é o Finance.AI, um assistente financeiro especialista.

REGRAS CRITICAS:
1. Para conceitos econômicos, definições e contexto (ex: Selic, Copom, Inflação, PIB), SEMPRE use 'search_documents'. NUNCA use 'redirect' para temas econômicos.
2. Para cotações e performance de ativos (ex: PETR4, NVDA, comparações), SEMPRE use 'get_stock_price' ou 'compare_stocks'.
3. Use 'redirect' APENAS para assuntos totalmente fora de finanças (ex: futebol, receitas, piadas).

Para perguntas conceituais sobre finanças, economia ou mercado financeiro, priorize sempre o uso de 'search_documents'.
Nunca responda diretamente sem utilizar as ferramentas de busca disponiveis.
"""
        
        # Create async agent engine
        self.agent = AsyncAgentEngine(
            llm=llm,
            registry=registry,
            executor=executor,
            system_instruction=system_instruction,
            tool_choice=None,
            skip_validation=skip_validation
        )
    
    async def run_stream(
        self,
        query: str,
        chat_history=None
    ) -> AsyncIterator[str]:
        """
        Run agent with async streaming.
        
        Args:
            query: User query
            chat_history: Optional chat history
            
        Yields:
            Response tokens as they arrive
        """
        # Run agent and get context
        from core.context_async import AsyncExecutionContext
        
        # Create context manually to have access to it
        context = AsyncExecutionContext(user_query=query)
        await context.set_memory("system_instruction", self.agent.system_instruction)
        await context.set_memory("chat_history", chat_history or [])
        
        # Run dispatch
        await self.agent.dispatch(context)
        
        # Store context for later access
        self.context = context
        
        # Collect answer for finalization
        answer = []
        
        # Stream from answer state
        if hasattr(self.agent.answer_state, 'generator') and self.agent.answer_state.generator:
            async for token in self.agent.answer_state.generator:
                answer.append(token)
                yield token
        
        # Finalize response with metadata
        final_answer = "".join(answer)
        await self._finalize_response(final_answer, context)
    
    async def _finalize_response(
        self,
        content: str,
        context
    ):
        """
        Calculate metrics and store in context memory
        """
        sources_used = []
        scores = []
        has_stock_data = False
        
        for call in context.tool_calls or []:
            tool_name = call.get("tool_name")
            result = call.get("result", {})
            
            if tool_name == "search_documents" and isinstance(result, dict):
                for doc in result.get("results", []):
                    meta = doc.get("metadata", {})
                    src = meta.get("source")
                    if src and src not in sources_used:
                        sources_used.append(src)
                    if "score" in doc:
                        scores.append(doc["score"])
            
            elif tool_name in ("get_stock_price", "compare_stocks"):
                if isinstance(result, dict) and result.get("success"):
                    sources_used.append(f"yfinance:{tool_name}")
                    has_stock_data = True
        
        confidence = self._calculate_confidence(
            has_stock_data,
            scores,
            sources_used
        )
        
        # Store in context.memory so API can access it
        await context.set_memory("sources_used", sources_used)
        await context.set_memory("confidence", confidence)
        await context.set_memory("final_answer", content)
    
    def _calculate_confidence(
        self,
        has_stock_data: bool,
        scores: list,
        sources_used: list
    ) -> str:
        if has_stock_data:
            return "high"
        
        if scores:
            avg_score = sum(scores) / len(scores)
            max_score = max(scores)
            
            if max_score > 0.7 or (avg_score > 0.6 and len(scores) >= 2):
                return "high"
            if avg_score >= 0.5:
                if max_score > 0.6:
                    return "medium"
                return "low"
            return "low"
        
        return "low" if not sources_used else "medium"
