from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field
from typing import Optional, Any
from datetime import datetime
from src.types import IntentType


class DocumentChunk(BaseModel):
    """Represents a chunk of document content"""
    doc_id: str = Field(description="Document identifier")
    content: str = Field(description="The actual text content")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    relevance_score: float = Field(default=0.0, description="Relevance score for retrieval")


class AnswerResponse(BaseModel):
    """Structured response for Q&A tasks"""
    question: str = Field(description="The original question")
    answer: str = Field(description="The generated answer")
    sources: list[str] = Field(default_factory=list, description="List of source document IDs")
    confidence: float = Field(description="Confidence score for the answer (0.0 to 1.0)")
    timestamp: datetime = Field(default_factory=datetime.now)


class SummarizationResponse(BaseModel):
    """Structured response for summarization tasks"""
    original_length: int = Field(description="Length of original text")
    summary: str = Field(description="The generated summary")
    key_points: list[str] = Field(description="List of key points extracted")
    document_ids: list[str] = Field(default_factory=list, description="Documents summarized")
    timestamp: datetime = Field(default_factory=datetime.now)


class CalculationResponse(BaseModel):
    """Structured response for calculation tasks"""
    expression: str = Field(description="The mathematical expression")
    result: float = Field(description="The calculated result")
    explanation: str = Field(description="Step-by-step explanation")
    units: Optional[str] = Field(default=None, description="Units if applicable")
    timestamp: datetime = Field(default_factory=datetime.now)


class UpdateMemoryResponse(BaseModel):
    """Response after updating memory"""
    summary: str = Field(description="Summary of the conversation up to this point")
    document_ids: list[str] = Field(default_factory=list, description="List of documents ids that are relevant to the users last message")


class UserIntent(BaseModel):
    """User intent classification"""
    intent_type: IntentType = Field(description="The classified intent")
    confidence: float = Field(description="Confidence score for the intent classification (0.0 to 1.0)")
    reasoning: str = Field(description="Explanation of how the intent was classified")


class SessionState(BaseModel):
    """Session state"""
    session_id: str
    user_id: str
    conversation_history: list[BaseMessage] = Field(default_factory=list)
    document_context: list[str] = Field(default_factory=list, description="Active document IDs")
    created_at: datetime = Field(default_factory=datetime.now)
    last_updated: datetime = Field(default_factory=datetime.now)
