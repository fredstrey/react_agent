"""
Tool executor with automatic Pydantic validation
"""
import json
from typing import Dict, Any, Optional
from pydantic import ValidationError
from .registry import ToolRegistry


class ToolExecutor:
    """Tool executor with automatic Pydantic validation"""
    
    def __init__(self, registry: Optional[ToolRegistry] = None):
        """
        Initialize executor
        
        Args:
            registry: Tool registry (uses singleton if not specified)
        """
        self.registry = registry or ToolRegistry()
    
    def execute(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a tool with automatic validation
        
        Args:
            tool_name: Name of the tool
            arguments: Arguments for the tool
            
        Returns:
            Execution result
        """
        # Search for tool
        tool_data = self.registry.get(tool_name)
        if not tool_data:
            raise ValueError(f"Tool '{tool_name}' not found")
        
        function = tool_data["function"]
        args_model = tool_data["args_model"]
        
        try:
            # Validate arguments with Pydantic
            validated_args = args_model(**arguments)
            
            # Execute function
            result = function(**validated_args.model_dump())
            
            # Return result
            return {
                "success": True,
                "result": result,
                "tool_name": tool_name
            }
            
        except ValidationError as e:
            # Validation error
            return {
                "success": False,
                "error": f"Validation error: {str(e)}",
                "tool_name": tool_name
            }
        
        except Exception as e:
            # Execution error
            return {
                "success": False,
                "error": f"Execution error: {str(e)}",
                "tool_name": tool_name
            }
    
    def execute_from_llm_response(self, llm_response: str) -> Optional[Dict[str, Any]]:
        """
        Extract and execute tool call from LLM response
        
        Args:
            llm_response: LLM response containing tool call
            
        Returns:
            Execution result or None if no tool call
        """
        tool_call = self._extract_tool_call(llm_response)
        if not tool_call:
            return None
        
        return self.execute(tool_call["name"], tool_call["arguments"])
    
    def _extract_tool_call(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Extract tool call from text
        
        Expected format: tool_name({"arg1": "value1", ...})
        """
        for tool_name in self.registry.list():
            if tool_name in text:
                try:
                    # Find JSON start
                    start = text.find(tool_name) + len(tool_name)
                    json_start = text.find("{", start)
                    if json_start == -1:
                        continue
                    
                    # Find JSON end
                    brace_count = 0
                    json_end = json_start
                    for i in range(json_start, len(text)):
                        if text[i] == '{':
                            brace_count += 1
                        elif text[i] == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                json_end = i + 1
                                break
                    
                    # Parse JSON
                    args_json = text[json_start:json_end]
                    arguments = json.loads(args_json)
                    
                    return {
                        "name": tool_name,
                        "arguments": arguments
                    }
                except:
                    continue
        
        return None
