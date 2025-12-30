"""
Decorator to register functions as tools
"""
import inspect
from typing import Callable, Optional, Any, get_type_hints
from functools import wraps
from pydantic import BaseModel, create_model, Field
from .registry import ToolRegistry


def tool(
    name: Optional[str] = None,
    description: Optional[str] = None
):
    """
    Decorator to register a function as a tool
    
    Usage:
        @tool(name="search", description="Search documents")
        def search_docs(query: str, limit: int = 3) -> SearchResult:
            ...
    
    Args:
        name: Name of the tool (uses function name if not specified)
        description: Description of the tool (uses docstring if not specified)
    """
    def decorator(func: Callable) -> Callable:
        # Tool name
        tool_name = name or func.__name__
        
        # Tool description
        tool_description = description or (func.__doc__ or "").strip()
        
        # Extract type hints
        type_hints = get_type_hints(func)
        sig = inspect.signature(func)
        
        # Create Pydantic schema automatically
        fields = {}
        for param_name, param in sig.parameters.items():
            if param_name == 'self':
                continue
            
            param_type = type_hints.get(param_name, Any)
            param_default = param.default if param.default != inspect.Parameter.empty else ...
            
            # Extract description from docstring if possible
            param_description = f"Parameter {param_name}"
            
            fields[param_name] = (param_type, Field(default=param_default, description=param_description))
        
        # Create Pydantic model dynamically
        ArgsModel = create_model(
            f"{tool_name.capitalize()}Args",
            **fields
        )
        
        # Return type
        return_type = type_hints.get('return', Any)
        
        # Register the tool
        registry = ToolRegistry()
        registry.register(
            name=tool_name,
            description=tool_description,
            function=func,
            args_model=ArgsModel,
            return_type=return_type
        )
        
        # Wrapper to maintain the original function
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        
        # Add metadata
        wrapper._tool_name = tool_name
        wrapper._tool_description = tool_description
        wrapper._args_model = ArgsModel
        
        return wrapper
    
    return decorator
