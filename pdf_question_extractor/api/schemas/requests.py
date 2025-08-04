from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator
from enum import Enum


class ProcessingStatusEnum(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing" 
    OCR_COMPLETE = "ocr_complete"
    LLM_COMPLETE = "llm_complete"
    EMBEDDING_COMPLETE = "embedding_complete"
    STORED = "stored"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class QuestionStatusEnum(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class QuestionTypeEnum(str, Enum):
    MULTIPLE_CHOICE = "multiple_choice"
    SHORT_ANSWER = "short_answer"
    ESSAY = "essay"
    CALCULATION = "calculation"
    DIAGRAM = "diagram"
    PRACTICAL = "practical"
    OTHER = "other"


class BulkOperationEnum(str, Enum):
    APPROVE = "approve"
    REJECT = "reject"
    DELETE = "delete"
    UPDATE_STATUS = "update_status"


class ExportFormatEnum(str, Enum):
    CSV = "csv"
    JSON = "json"
    EXCEL = "excel"


# Request Models

class UploadRequest(BaseModel):
    """Request model for file upload"""
    store_to_db: bool = Field(default=True, description="Whether to store extracted questions to database")
    generate_embeddings: bool = Field(default=True, description="Whether to generate vector embeddings")
    recursive: bool = Field(default=True, description="For folder uploads, whether to process subdirectories")
    max_concurrent: int = Field(default=3, ge=1, le=10, description="Maximum concurrent file processing")


class QuestionUpdateRequest(BaseModel):
    """Request model for updating a single question"""
    question_number: Optional[str] = Field(None, max_length=50)
    marks: Optional[int] = Field(None, ge=0, le=1000)
    year: Optional[str] = Field(None, max_length=10)
    level: Optional[str] = Field(None, max_length=50)
    topics: Optional[List[str]] = Field(None, max_items=20)
    question_type: Optional[QuestionTypeEnum] = None
    question_text: Optional[str] = Field(None, min_length=1)
    status: Optional[QuestionStatusEnum] = None
    extra_metadata: Optional[Dict[str, Any]] = None

    @validator('topics')
    def validate_topics(cls, v):
        if v is not None:
            return [topic.strip() for topic in v if topic.strip()]
        return v


class BulkOperationRequest(BaseModel):
    """Request model for bulk operations on questions"""
    question_ids: List[int] = Field(..., min_items=1, description="List of question IDs to operate on")
    operation: BulkOperationEnum = Field(..., description="Type of bulk operation")
    new_status: Optional[QuestionStatusEnum] = Field(None, description="New status for update_status operation")
    update_data: Optional[QuestionUpdateRequest] = Field(None, description="Data to apply for bulk updates")

    @validator('new_status')
    def validate_new_status_with_operation(cls, v, values):
        if values.get('operation') == BulkOperationEnum.UPDATE_STATUS and v is None:
            raise ValueError("new_status is required for update_status operation")
        return v


class SaveApprovedRequest(BaseModel):
    """Request model for saving approved questions to permanent storage"""
    question_ids: Optional[List[int]] = Field(None, description="Specific question IDs to save (if None, saves all approved)")
    clear_extracted: bool = Field(default=True, description="Whether to clear extracted questions after saving")


class ExportRequest(BaseModel):
    """Request model for exporting questions"""
    format: ExportFormatEnum = Field(default=ExportFormatEnum.CSV, description="Export format")
    status_filter: Optional[QuestionStatusEnum] = Field(None, description="Filter by question status")
    year_filter: Optional[str] = Field(None, description="Filter by year")
    level_filter: Optional[str] = Field(None, description="Filter by level")
    question_type_filter: Optional[QuestionTypeEnum] = Field(None, description="Filter by question type")
    topics_filter: Optional[List[str]] = Field(None, description="Filter by topics (AND logic)")
    source_pdf_filter: Optional[str] = Field(None, description="Filter by source PDF filename")
    include_metadata: bool = Field(default=True, description="Whether to include metadata columns")
    question_ids: Optional[List[int]] = Field(None, description="Specific question IDs to export")

    @validator('topics_filter')
    def validate_topics_filter(cls, v):
        if v is not None:
            return [topic.strip() for topic in v if topic.strip()]
        return v


class SearchRequest(BaseModel):
    """Request model for searching questions"""
    query: str = Field(..., min_length=1, max_length=500, description="Search query text")
    semantic_search: bool = Field(default=True, description="Whether to use semantic vector search")
    status_filter: Optional[QuestionStatusEnum] = Field(None, description="Filter by question status")
    year_filter: Optional[str] = Field(None, description="Filter by year")
    level_filter: Optional[str] = Field(None, description="Filter by level")
    question_type_filter: Optional[QuestionTypeEnum] = Field(None, description="Filter by question type")
    topics_filter: Optional[List[str]] = Field(None, description="Filter by topics")
    limit: int = Field(default=20, ge=1, le=100, description="Maximum number of results")
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="Minimum similarity score for semantic search")


class WebSocketMessage(BaseModel):
    """WebSocket message structure"""
    type: str = Field(..., description="Message type")
    data: Optional[Dict[str, Any]] = Field(None, description="Message data")
    client_id: Optional[str] = Field(None, description="Client identifier")
    timestamp: Optional[str] = Field(None, description="Message timestamp")