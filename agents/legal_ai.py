"""
Legal.AI Agent
=================

Specialized legal consultant assistant using the high-level Agent API.

This demonstrates how to create a domain-specific agent with:
- Custom persona (Legal.AI)
- Domain-specific tools (document search)
- Parallel execution for complex queries
- Intent analysis for efficient routing
"""

import os
import sys
from datetime import datetime
from typing import AsyncIterator

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from finitestatemachineAgent import Agent
from agents import legal_ai_utils, legal_ai_tools
from agents.legal_ai_utils import LegalEpistemicContractStrategy

class LegalAI:
    """
    Legal.AI - Specialized Legal Consultant
    
    Built on top of the HFSM Agent framework with:
    - Document search for legal concepts
    - Parallel research capabilities
    - Intent-based routing
    """
    
    def __init__(
        self,
        llm_provider: str = "openrouter",
        model: str = "xiaomi/mimo-v2-flash:free",
        api_key: str = None,
        embedding_manager = None
    ):
        """
        Initialize Legal.AI agent.
        
        Args:
            llm_provider: LLM provider (default: openrouter)
            model: Model to use
            api_key: API key (defaults to OPENROUTER_API_KEY env var)
            embedding_manager: Optional EmbeddingManager instance
        """
        # Initialize tools if manager is provided
        if embedding_manager:
            legal_ai_tools.initialize_rag_tools(embedding_manager)

        # Define Legal.AI Persona
        self.persona = """Você é o **Legal.AI**, um Consultor Jurídico Sênior de alta especialização.

OBJETIVO:
Fornecer análises jurídicas com VERNÁCULO CULTO, RIGOR TÉCNICO e PRECISÃO DOUTRINÁRIA.

FERRAMENTAS DISPONÍVEIS:
- **search_documents**: Ferramenta de busca em base de dados jurídica com fontes oficiais e validadas (Constituição Federal, Códigos, Leis, Jurisprudência, Doutrina).

**USO OBRIGATÓRIO DA FERRAMENTA:**
1. Para QUALQUER pergunta jurídica que exija fundamentação legal, você DEVE utilizar a ferramenta search_documents ANTES de responder.
2. **PROIBIDO**: Jamais responda questões jurídicas baseando-se apenas em conhecimento geral sem consultar a ferramenta. A precisão e a citação de fontes oficiais são IMPERATIVAS.

**MÚLTIPLAS BUSCAS PARA QUESTÕES COMPLEXAS:**
3. **Se a pergunta envolver MÚLTIPLOS TÓPICOS ou ASPECTOS JURÍDICOS diferentes**, você DEVE chamar a ferramenta search_documents MÚLTIPLAS VEZES, uma para cada tópico/aspecto.
4. **EXECUÇÃO PARALELA**: Você pode (e deve) chamar VÁRIAS ferramentas AO MESMO TEMPO. O sistema executa todas as buscas em paralelo para maior eficiência.
5. **Exemplo**: Para "Quais são os direitos do consumidor e as responsabilidades do fornecedor?", faça DUAS buscas:
   - search_documents(query="direitos do consumidor CF/88 CDC")
   - search_documents(query="responsabilidades fornecedor CDC")

**ESTRATÉGIA DE BUSCA:**
- Para questões amplas: Divida em sub-tópicos e busque cada um separadamente
- Para comparações: Busque cada elemento comparado individualmente
- Para análises multi-dimensionais: Busque cada dimensão (constitucional, legal, doutrinária, jurisprudencial)

DIRETRIZES DE ESTILO E TOM (JURIDIQUÊS OBRIGATÓRIO):
1. **Linguagem**: Empregue estritamente o "juridiquês" culto. Utilize vocabulário técnico-jurídico preciso (ex: "exordial", "lide", "fomus boni iuris", "periculum in mora").
2. **Latinismos**: Utilize brocardos e expressões latinas pertinentes (ex: *data venia*, *in dubio pro reo*, *pacta sunt servanda*, *erga omnes*) para enriquecer a argumentação.
3. **Formalidade**: Mantenha tom solene, impessoal e distanciado. Evite coloquialismos ou linguagem simplificada. Trate o interlocutor com a devida vênia.
4. **Estrutura**: Organize o raciocínio em silogismos jurídicos (premissa maior, premissa menor, conclusão).

ESCOPO DE ATUAÇÃO:
- Análise de leis e códigos (Constituição, Código Civil, Código Penal, etc.).
- Interpretação de Jurisprudência e Súmulas.
- Dissecação de cláusulas contratuais.
- Doutrina jurídica.

LIMITES ÉTICOS E TÉCNICOS:
- **Não Inventar**: Jamais fabrique leis ou citações. Se a informação não constar na busca ou em seu conhecimento treinado, declare a ausência de amparo.
- **Natureza do Agente**: Identifique-se como Inteligência Artificial de apoio jurídico. Deixe claro que a orientação não substitui o patrocínio de um advogado regularmente inscrito na OAB para o contencioso.

Sua resposta deve ser uma PEÇA JURÍDICA concisa e fundamentada."""

        # Define Redirect Prompt (for simple queries)
        self.redirect_prompt = f"""Você é o **Legal.AI**, um Consultor Jurídico especializado.
Data de hoje: {datetime.now().strftime('%d/%m/%Y')}

DIRETRIZES DE RESPOSTA RÁPIDA:
1. **Saudações**: Responda formalmente (ex: "Bom dia. Em que posso auxiliá-lo juridicamente hoje?").
2. **Escopo**: Foco em questões jurídicas, análise de documentos e conceitos legais.
3. **Fora de Escopo**: Se a pergunta não for jurídica, redirecione educadamente para o tema jurídico ou encerre.
4. Seja conciso e direto."""

        # Initialize Agent with Legal.AI configuration
        self.agent = Agent(
            # LLM Config
            llm_provider=llm_provider,
            model=model,
            api_key=api_key,
            
            # Persona
            system_instruction=self.persona,
            redirect_prompt=self.redirect_prompt,
            
            # Legal Tools (RAG only)
            tools=[
                legal_ai_tools.search_documents
            ],
            
            # Features

            # enable forks for parallel research
            enable_parallel_planning=False,

            # enable planning, intent analysis and creates ToDo list for tasks
            enable_intent_analysis=True, 

            # maximum number of parallel branches when using forks
            max_parallel_branches=3,
            
            # Safety
            max_global_requests=50,

            # after tool is used, goes to validation state to validate the result
            skip_validation=False,
            
            # Legal.AI-specific customizations

            # custom tools validations
            # if no function is defined, it will use llm to validates as default
            validation_fn=legal_ai_utils.tools_validation,
            
            # custom planning system prompt
            # you can use this to override the basic prompt for parallel planning using your own prompt
            planning_system_prompt=legal_ai_utils.enhance_rag_planning_prompt,
            
            # custom post router hook, forces tool usage after validation node
            # you can use this after router decision, intercept and alter router decision (ex: enforce tool usage)     
            post_router_hook=legal_ai_utils.enforce_tool_usage,

            contract_strategy="epistemic",   # contract strategy for forks (simple, epistemic, your-custom-strategy)
            synthesis_strategy="llm", # synthesis strategy for forks (llm, concat, your-custom-strategy)

        )

        # Hot-swap contract strategy for LegalAI specialization
        self.agent.engine.contract_strategy = LegalEpistemicContractStrategy(self.agent.llm)
    
    async def run(self, query: str, chat_history: list = None):
        """
        Execute a query and return the complete response.
        
        Args:
            query: User query
            chat_history: Optional chat history
            
        Returns:
            AgentResponse with content and metadata
        """
        return await self.agent.run(query, chat_history)
    
    async def stream(self, query: str, chat_history: list = None) -> AsyncIterator[str]:
        """
        Stream response tokens as they are generated.
        
        Args:
            query: User query
            chat_history: Optional chat history
            
        Yields:
            Response tokens
        """
        async for token in self.agent.stream(query, chat_history):
            yield token
        
        # Extract metadata after streaming completes
        if hasattr(self.agent, 'last_context'):
            await legal_ai_utils.extract_metadata(self.agent.last_context)