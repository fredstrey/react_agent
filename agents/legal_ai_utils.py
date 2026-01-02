"""
Legal.AI Utilities
=====================

Helper functions and configurations for the Legal.AI agent.
Includes validation, planning enhancements, and hooks.
"""

import logging
from finitestatemachineAgent.hfsm_agent_async import Transition
import json
from typing import Any
from core.context_async import AsyncExecutionContext
from finitestatemachineAgent.contract_strategies import ForkContractStrategy
from finitestatemachineAgent.fork_contracts import ForkResult, Claim, EvidenceType

logger = logging.getLogger(__name__)


async def tools_validation(context, tool_name, result):
    """
    Custom validation logic for Legal.AI tools.
    
    Args:
        context: Execution context
        tool_name: Name of the tool that was executed
        result: Result returned by the tool
        
    Returns:
        True if the tool result is valid, False otherwise
    """
    if tool_name in ("get_stock_price", "compare_stocks"):
        # For stock tools, check if result has success=True
        return isinstance(result, dict) and result.get("success") == True
    
    elif tool_name == "search_documents":
        # For document search, check if we have results
        return isinstance(result, dict) and result.get("results") and len(result.get("results", [])) > 0
    
    # Default: accept any non-None result
    return result is not None


def enhance_rag_planning_prompt(default_prompt, context):
    """
    Enhances default planning prompt with Legal.AI-specific divide-and-conquer strategy.
    
    Instructs LLM to break complex legal queries into smaller independent tasks
    for parallel execution.
    
    Args:
        default_prompt: Default planning prompt from engine
        context: Execution context
        
    Returns:
        Enhanced prompt with RAG-specific instructions
    """
    enhancement = """

ESTRATÉGIA DIVIDIR E CONQUISTAR (LEGAL.AI):

Para consultas legais complexas, você DEVE quebrar em sub-tarefas independentes focadas em encontrar evidências:

0. **Análise Constitucional (PRIORIDADE MÁXIMA)**:
   - VERIFIQUE SEMPRE a compatibilidade constitucional primeiro.
   - Crie um branch para buscar princípios e normas na Constituição Federal de 1988.
   - Termos de busca obrigatórios: "CF/88", "Constituição Federal", "Princípios Constitucionais", "Direitos Fundamentais".
   - Ex: "Verificar constitucionalidade do tema X na CF/88"

1. **Identificação de Fontes Legais (Infraconstitucional)**:
   - Se a pergunta requer base legal (leis, artigos, jurisprudência)
   - Crie um branch específico para buscar legislações e artigos relevantes
   - Ex: "Buscar artigos do Código Civil sobre X", "Buscar Tese do STJ sobre Y"

2. **Análise de Claims (Alegações)**:
   - Identifique os pontos centrais da questão
   - Crie branches para validar cada alegação com documentos/provas
   - Ex: "Buscar evidências que sustentam a alegação de dano moral"

3. **Contrafictions (Contorversas/Contra-argumentos)**:
   - Crie um branch dedicado a buscar posições contrárias, exceções ou jurisprudência divergente
   - Essencial para uma análise jurídica robusta e imparcial
   - Ex: "Buscar excludentes de ilicitude para o caso", "Buscar jurisprudência contrária ao pedido"

4. **Comparações de Documentos**:
   - Se comparar documentos A e B, mantenha branches separados para cada um

REGRAS IMPORTANTES:
- Objetivo é municiar a resposta final com: FUNDAMENTAÇÃO LEGAL, EVIDÊNCIAS DE SUPORTE e PONTOS DE ATENÇÃO (CONTRA).
- Máximo de 3 branches.
- Cada branch deve ter um objetivo de BUSCA claro.

EXEMPLOS:

Query: "Analise a viabilidade de uma ação de usucapião neste caso..."
→ strategy: "parallel_research"
→ branches: [
    {"id": "lei", "goal": "Buscar requisitos legais da usucapião no Código Civil"},
    {"id": "juris", "goal": "Buscar jurisprudência recente sobre usucapião urbano"},
    {"id": "contra", "goal": "Buscar impedimentos e casos de improcedência de usucapião"}
]

Query: "Compare as cláusulas de rescisão do Contrato A e Contrato B"
→ strategy: "parallel_research"
→ branches: [
    {"id": "doc_a", "goal": "Buscar cláusula de rescisão no Contrato A"},
    {"id": "doc_b", "goal": "Buscar cláusula de rescisão no Contrato B"}
]
    
Query: "O que a Constituição diz sobre liberdade de expressão?"
→ strategy: "parallel_research"
→ branches: [
    {"id": "cf88", "goal": "Buscar artigos sobre liberdade de expressão na Constituição Federal de 1988"}
]"""
    
    return default_prompt + enhancement


async def enforce_tool_usage(context, transition):
    """
    Finance.AI-specific hook: Reject direct answers, force tool usage.
    
    This keeps the engine domain-agnostic while allowing Finance.AI
    to enforce its own rules about always using tools for financial data.
    
    Args:
        context: Execution context
        transition: Proposed transition
        
    Returns:
        Modified transition if needed, or None to keep original
    """
    # Skip enforcement in fork contexts (forks have their own flow)
    is_fork = await context.get_memory("branch_id") is not None
    if is_fork:
        return None
    
    # Check if IntentAnalysis classified this as a simple query
    intent_analysis = await context.get_memory("intent_analysis", {})
    complexity = intent_analysis.get("complexity", "simple")
    needs_tools = intent_analysis.get("needs_tools", False)
    
    # Skip enforcement for simple queries that don't need tools
    if complexity == "simple" and not needs_tools:
        logger.info("🚀 [Finance.AI] Allowing direct answer for simple query")
        return None
    
    if transition.to == "AnswerState" and transition.reason == "Direct answer generation":
        # LLM tried to answer directly without tools - unacceptable for Finance.AI
        retry_count = await context.get_memory("rag_tool_retry", 0)
        
        if retry_count < 2:
            await context.set_memory("rag_tool_retry", retry_count + 1)
            logger.info(f"🔄 [Finance.AI] Forcing tool usage (attempt {retry_count + 1}/2)")
            
            # Override transition to retry
            return Transition(to="RetryState", reason="Finance.AI requires tool usage")
        else:
            logger.error("❌ [Finance.AI] LLM refusing to use tools after retries")
            # Let it fail to RetryState
            return Transition(to="RetryState", reason="Tool usage required")
    
    # Reset retry count on successful tool usage
    if transition.to == "ToolState":
        await context.set_memory("rag_tool_retry", 0)
    
    return None  # Keep original transition


async def extract_metadata(context):
    """
    Extract sources_used and confidence from tool results.
    
    This is Finance.AI-specific logic for metadata extraction.
    Should be called after answer generation to populate metadata.
    
    Args:
        context: Execution context with tool_calls
        
    Returns:
        dict with sources_used and confidence
    """
    sources_used = []
    has_data = False
    
    # Include merged tool calls from parallel execution
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
            has_data = True

    
    # Calculate confidence
    confidence = "high" if has_data else "low"
    
    # Store in context memory for API access
    await context.set_memory("sources_used", sources_used)
    await context.set_memory("confidence", confidence)
    
    return {
        "sources_used": sources_used,
        "confidence": confidence
    }


class LegalEpistemicContractStrategy(ForkContractStrategy):
    """
    Legal-specific strategy: Extracts Articles, Jurisprudence, and Legal Claims.
    """
    
    async def extract(self, context: AsyncExecutionContext, branch_id: str, branch_goal: str) -> ForkResult:
        logger.info(f"⚖️ [LegalContract] Extracting legal data for branch: {branch_id}")
        
        research_notes = await context.get_memory("research_notes", "")
        
        # Build legal claim extraction prompt
        system_instruction = """You are a Legal Research Assistant.

Your job:
1. Analyze research results from legal documents.
2. Extract VERIFIABLE LEGAL PROVISIONS (Articles, Laws) and JURISPRUDENCE.
3. Extract legal claims/arguments supported by these documents.

Output ONLY valid JSON in this exact format:
{
  "claims": [
    {
        "key": "lei.artigo", 
        "value": "Art. X da Lei Y: texto...", 
        "evidence": [{"type": "retrieved", "source": "tool:search_documents"}], 
        "confidence": 0.9
    },
    {
        "key": "constituicao.principio", 
        "value": "Princípio da Dignidade da Pessoa Humana (Art. 1º, III, CF/88)...", 
        "evidence": [{"type": "retrieved", "source": "tool:search_documents"}], 
        "confidence": 0.95
    },
    {
        "key": "constituicao.norma", 
        "value": "Art. 5º, X, CF/88: são invioláveis a intimidade...", 
        "evidence": [{"type": "retrieved", "source": "tool:search_documents"}], 
        "confidence": 0.95
    },
    {
        "key": "jurisprudencia.conteudo", 
        "value": "Súmula X do STJ: texto...", 
        "evidence": [{"type": "retrieved", "source": "tool:search_documents"}], 
        "confidence": 0.9
    },
    {
        "key": "argumento.juridico", 
        "value": "A conduta configura dano moral in re ipsa...", 
        "evidence": [{"type": "inferred", "source": "analysis"}], 
        "confidence": 0.7
    }
  ],
  "coverage": ["topic1", "topic2"],
  "uncertain_topics": [{"topic": "topic3", "reason": "law_not_found"}]
}

IMPORTANT:
- Keys should be hierarchical: "constituicao.principio", "constituicao.norma", "lei.civil", "jurisprudencia.stj".
- PRIORITIZE CONSTITUTIONAL NORMS (CF/88). Identify principles explicitly.
- Values must be EXPLICIT text from laws when possible.
- DO NOT invent laws."""
        
        # Prepare tool results
        tool_results_text = ""
        if context.tool_calls:
            for call in context.tool_calls:
                tool_name = call.get("tool_name", "unknown")
                result = call.get("result", "")
                tool_results_text += f"\nTool: {tool_name}\nResult: {str(result)[:800]}\n" # Expanded context for legal text
        
        if research_notes:
            tool_results_text += f"\nDirect notes: {research_notes}\n"
        
        # Combine into single user message
        user_content = f"Research goal: {branch_goal}\n\nLegal Evidence:\n{tool_results_text}\n\nExtract legal claims as JSON:"
        
        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_content}
        ]
        
        # Call LLM
        response = await self.llm.chat(messages)
        content = response.get("content", "").strip()
        
        try:
            # Parse JSON
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            
            contract_data = json.loads(content.strip())
            
            # Map to Pydantic
            claims = [Claim(**c) for c in contract_data.get("claims", [])]
            
            return ForkResult(
                branch_id=branch_id,
                claims=claims,
                coverage=contract_data.get("coverage", []),
                uncertain_topics=contract_data.get("uncertain_topics", []),
                confidence=sum(c.confidence for c in claims) / len(claims) if claims else 0.0
            )

        except Exception as e:
            logger.error(f"❌ [LegalContract] Failed: {e}")
            return ForkResult(branch_id=branch_id, confidence=0.0)
