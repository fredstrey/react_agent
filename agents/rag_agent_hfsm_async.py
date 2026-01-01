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

REGRA ANTI-REDUNDÂNCIA (CRÍTICO):
4. ANTES de chamar qualquer ferramenta, VERIFIQUE se você já tem os dados necessários nas chamadas de ferramentas anteriores (tool calls).
5. Se você já chamou uma ferramenta e recebeu os dados, NÃO chame a mesma ferramenta novamente com os mesmos parâmetros.
6. Use os resultados das ferramentas já executadas para responder a pergunta. Só chame uma nova ferramenta se realmente precisar de informações adicionais diferentes.

Para perguntas conceituais sobre finanças, economia ou mercado financeiro, priorize sempre o uso de 'search_documents'.
Nunca responda diretamente sem utilizar as ferramentas de busca disponiveis.
"""
        
        # Custom validation function for RAG tools
        async def rag_validation(context, tool_name, result):
            """
            Custom validation logic for RAG agent tools.
            Returns True if the tool result is valid, False otherwise.
            """
            if tool_name in ("get_stock_price", "compare_stocks"):
                # For stock tools, check if result has success=True
                return isinstance(result, dict) and result.get("success") == True
            
            elif tool_name == "search_documents":
                # For document search, check if we have results
                return isinstance(result, dict) and result.get("results") and len(result.get("results", [])) > 0
            
            elif tool_name == "redirect":
                # Redirect always succeeds if it returns a result
                return result is not None
            
            # Default: accept any non-None result
            return result is not None
        
        # Custom planning prompt enhancer for parallel execution
        def enhance_rag_planning_prompt(default_prompt, context):
            """
            Enhances default planning prompt with RAG-specific divide-and-conquer strategy.
            Instructs LLM to break complex financial queries into smaller independent tasks.
            """
            # Add RAG-specific planning instructions
            enhancement = """

ESTRATÉGIA DIVIDIR E CONQUISTAR (RAG AGENT):

Para consultas financeiras complexas, você DEVE quebrar em sub-tarefas independentes:

1. **Comparações de Ativos**: 
   - Se comparar múltiplos ativos (ex: "Compare PETR4, VALE3 e ITUB4")
   - Crie um branch para cada ativo
   - Cada branch pesquisa um ativo específico

2. **Análises Multi-Tópico**:
   - Se a pergunta envolve múltiplos conceitos (ex: "Explique Selic, Copom e inflação")
   - Crie um branch para cada conceito
   - Cada branch pesquisa um conceito específico

3. **Consultas Compostas**:
   - Se combina dados + conceitos (ex: "Qual o preço do PETR4 e o que é dividend yield?")
   - Branch 1: Buscar preço do ativo
   - Branch 2: Buscar conceito teórico

REGRAS IMPORTANTES:
- Só paralelizar se as sub-tarefas forem INDEPENDENTES
- Cada branch deve ter um objetivo claro e específico
- Máximo de 3 branches.
- Para consultas simples (1 ativo, 1 conceito), use strategy: "single"

EXEMPLOS:

Query: "Compare NVDA e TSLA"
→ strategy: "parallel_research"
→ branches: [
    {"id": "nvda", "goal": "Pesquisar preço e dados da NVDA"},
    {"id": "tsla", "goal": "Pesquisar preço e dados da TSLA"}
]

Query: "Qual o preço do PETR4?"
→ strategy: "single" (consulta simples, não precisa paralelizar)

Query: "Explique Selic, Copom e CDI"
→ strategy: "parallel_research"
→ branches: [
    {"id": "selic", "goal": "Pesquisar conceito de Selic"},
    {"id": "copom", "goal": "Pesquisar conceito de Copom"},
    {"id": "cdi", "goal": "Pesquisar conceito de CDI"}
]"""
            
            return default_prompt + enhancement
        
        # Create async agent engine with custom validation and parallel execution
        self.agent = AsyncAgentEngine(
            llm=llm,
            registry=registry,
            executor=executor,
            system_instruction=system_instruction,
            tool_choice=None,
            skip_validation=skip_validation,
            validation_fn=rag_validation,  # Custom validation
            
            # Enable parallel execution with custom planning
            enable_parallel_planning=True,
            planning_system_prompt=enhance_rag_planning_prompt,  # Incremental enhancement
            # merge_fn=None -> uses default append merge
            max_parallel_branches=3    # 🔥 Limit width to 3 branches per fork
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
        
        # Stream from context memory (not from state instance)
        stream = await context.get_memory("answer_stream")
        if stream:
            async for token in stream:
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
        
        has_stock_data = False
        
        # 🔥 NEW: include merged tool calls from parallel execution
        merged_tools = await context.get_memory("merged_tool_calls", [])
        all_calls = (context.tool_calls or []) + merged_tools
        
        for call in all_calls:
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
