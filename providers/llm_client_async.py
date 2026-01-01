"""
Async LLM Client
=================

Async wrapper for LLM providers.
"""

from typing import List, Dict, AsyncIterator
from providers.openrouter_async import AsyncOpenRouterProvider


class AsyncLLMClient:
    """
    Async adapter for LLM providers.
    
    Provides unified async interface for different LLM providers.
    """
    
    def __init__(
        self,
        model: str,
        temperature: float = 0.3,
    ):
        """
        Initialize async LLM client.
        
        Args:
            model: Model name
            temperature: Sampling temperature
        """
        self.provider = AsyncOpenRouterProvider(
            model=model,
            temperature=temperature
        )
    
    async def chat(self, messages: List[Dict[str, str]]) -> Dict:
        """
        Async chat completion.
        
        Args:
            messages: List of message dicts
            
        Returns:
            Dict with 'content' and 'usage'
        """
        return await self.provider.chat(messages)
    
    async def chat_with_tools(
        self,
        messages: List[Dict[str, str]],
        tools: List[Dict],
        tool_choice: str = None
    ) -> Dict:
        """
        Async chat with function calling.
        
        Args:
            messages: List of message dicts
            tools: Tool definitions
            tool_choice: Optional tool choice ("auto", "required", "none")
            
        Returns:
            Dict with 'content', 'tool_calls', and 'usage'
        """
        return await self.provider.chat_with_tools(messages, tools, tool_choice)
    
    async def chat_stream(
        self,
        messages: List[Dict[str, str]],
        context=None
    ) -> AsyncIterator[str]:
        """
        Async streaming chat.
        
        Args:
            messages: List of message dicts
            context: Optional execution context for usage tracking
            
        Yields:
            Response tokens
        """
        async for token in self.provider.chat_stream(messages):
            # Check if token is usage metadata (dict with __usage__ key)
            if isinstance(token, dict) and "__usage__" in token:
                # Accumulate usage to context if provided
                if context:
                    await context.accumulate_usage(token["__usage__"])
            else:
                # Regular content token
                yield token

