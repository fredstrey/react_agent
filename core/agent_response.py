"""
Agent Response Model
"""
from pydantic import BaseModel, Field


class AgentResponse(BaseModel):
    """Standardized agent response"""
    answer: str = Field(..., description="Final agent response")
