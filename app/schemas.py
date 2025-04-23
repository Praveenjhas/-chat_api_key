from pydantic import BaseModel
from typing import List

class BotRequest(BaseModel):
    """Request model for asking questions"""
    question: str

class BotResponse(BaseModel):
    """Response model for bot answers with sources"""
    question: str
    answer: str
    sources: List[str]  # Added sources field