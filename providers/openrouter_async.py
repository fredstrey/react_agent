"""
OpenRouter Provider - Async Version
====================================

Async implementation using httpx for better concurrency.
"""

import os
import httpx
import json
from typing import List, Dict, Optional, AsyncIterator


class AsyncOpenRouterProvider:
    """
    Async LLM provider using OpenRouter API with httpx.
    
    Supports:
    - Async chat completion
    - Async streaming responses
    - Concurrent requests
    """
    
    def __init__(
        self,
        model: str = "xiaomi/mimo-v2-flash:free",
        api_key: Optional[str] = None,
        base_url: str = "https://openrouter.ai/api/v1",
        temperature: float = 0.3,
        timeout: float = 30.0
    ):
        """
        Initialize async OpenRouter provider.
        
        Args:
            model: Model name
            api_key: OpenRouter API key
            base_url: API base URL
            temperature: Sampling temperature
            timeout: Request timeout in seconds
        """
        self.model = model
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenRouter API key required. "
                "Set OPENROUTER_API_KEY env var or pass api_key parameter."
            )
        
        self.base_url = base_url
        self.temperature = temperature
        self.timeout = timeout
    
    async def chat(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict]] = None
    ) -> Dict:
        """
        Async chat completion.
        
        Args:
            messages: List of message dicts
            tools: Optional tool definitions
            
        Returns:
            Dict with 'content' and 'usage'
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "stream": False
        }
        
        if tools:
            payload["tools"] = tools
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/fredstrey/react_agent",
            "X-Title": "Finance.AI"
        }
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=headers
            )
            
            if not response.is_success:
                print(f"OpenRouter Error: {response.text}")
            
            response.raise_for_status()
            
            data = response.json()
            return {
                "content": data["choices"][0]["message"]["content"],
                "usage": data.get("usage", {})
            }
    
    async def chat_with_tools(
        self,
        messages: List[Dict[str, str]],
        tools: List[Dict],
        tool_choice: Optional[str] = None
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
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "tools": tools,
            "stream": False
        }
        
        # Only add tool_choice if explicitly set (not None)
        # Some models don't support this parameter
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/fredstrey/react_agent",
            "X-Title": "Finance.AI"
        }
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=headers
            )
            response.raise_for_status()
            
            data = response.json()
            message = data["choices"][0]["message"]
            
            return {
                "content": message.get("content", ""),
                "tool_calls": message.get("tool_calls"),
                "usage": data.get("usage", {})
            }
    
    async def chat_stream(
        self,
        messages: List[Dict[str, str]]
    ) -> AsyncIterator[str]:
        """
        Async streaming chat completion.
        
        Args:
            messages: List of message dicts
            
        Yields:
            Response tokens as they arrive
            
        Note:
            Usage information is sent as the last chunk with a special marker.
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "stream": True,
            "stream_options": {"include_usage": True}  # Request usage in stream
        }
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/fredstrey/react_agent",
            "X-Title": "Finance.AI"
        }
        
        usage_data = None
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=headers
            ) as response:
                response.raise_for_status()
                
                async for line in response.aiter_lines():
                    if line and line.startswith("data: "):
                        data_str = line[6:]  # Remove "data: " prefix
                        
                        if data_str == "[DONE]":
                            break
                        
                        try:
                            data = json.loads(data_str)
                            
                            # Check for usage information
                            if "usage" in data:
                                usage_data = data["usage"]
                            
                            # Yield content tokens
                            delta = data["choices"][0]["delta"]
                            if "content" in delta:
                                yield delta["content"]
                        except json.JSONDecodeError:
                            continue
        
        # Yield usage as a special marker (dict instead of string)
        if usage_data:
            yield {"__usage__": usage_data}

