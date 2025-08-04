from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field
from datetime import datetime
from .requests import ProcessingStatusEnum, QuestionStatusEnum, QuestionTypeEnum


# Response Models

class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(..., description="Current timestamp")
    version: str = Field(..., description="API version")
    database_connected: bool = Field(..., description="Database connection status")
    services: Dict[str, str] = Field(..., description="Individual service statuses")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ProcessingProgressResponse(BaseModel):
    """Processing progress response"""
    file_path: str = Field(..., description="Path to the file being processed")
    status: ProcessingStatusEnum = Field(..., description="Current processing status")
    current_step: str = Field(..., description="Current processing step")
    total_steps: int = Field(..., description="Total number of processing steps")
    completed_steps: int = Field(..., description="Number of completed steps")
    progress_percentage: float = Field(..., description="Progress as percentage")
    start_time: datetime = Field(..., description="Processing start time")
    end_time: Optional[datetime] = Field(None, description="Processing end time")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    questions_extracted: int = Field(default=0, description="Number of questions extracted")
    questions_stored: int = Field(default=0, description="Number of questions stored")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
    embeddings_generated: int = Field(default=0, description="Number of embeddings generated")
    is_complete: bool = Field(..., description="Whether processing is complete")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class QuestionResponse(BaseModel):
    """Single question response"""
    id: int = Field(..., description="Question ID")
    question_number: Optional[str] = Field(None, description="Question number")
    marks: Optional[int] = Field(None, description="Question marks")
    year: Optional[str] = Field(None, description="Exam year")
    level: Optional[str] = Field(None, description="Education level")
    topics: List[str] = Field(default_factory=list, description="Question topics")
    question_type: Optional[str] = Field(None, description="Question type")
    question_text: str = Field(..., description="Question text content")
    source_pdf: str = Field(..., description="Source PDF filename")
    status: str = Field(..., description="Question status")
    modified: bool = Field(default=False, description="Whether question was modified")
    extraction_date: Optional[datetime] = Field(None, description="Date extracted")
    created_at: Optional[datetime] = Field(None, description="Date created")
    updated_at: Optional[datetime] = Field(None, description="Date last updated")
    extra_metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None
        }


class QuestionsListResponse(BaseModel):
    """Paginated questions list response"""
    questions: List[QuestionResponse] = Field(..., description="List of questions")
    total: int = Field(..., description="Total number of questions")
    page: int = Field(..., description="Current page number")
    per_page: int = Field(..., description="Items per page")
    total_pages: int = Field(..., description="Total number of pages")
    has_next: bool = Field(..., description="Whether there is a next page")
    has_prev: bool = Field(..., description="Whether there is a previous page")


class UploadResponse(BaseModel):
    """File upload response"""
    success: bool = Field(..., description="Whether upload was successful")
    message: str = Field(..., description="Response message")
    processing_id: Optional[str] = Field(None, description="Processing job ID")
    files_uploaded: List[str] = Field(default_factory=list, description="List of uploaded file paths")
    total_files: int = Field(default=0, description="Total number of files to process")


class ProcessingResultResponse(BaseModel):
    """Single file processing result"""
    success: bool = Field(..., description="Whether processing was successful")
    file_path: str = Field(..., description="Path to processed file")
    questions: List[Dict[str, Any]] = Field(default_factory=list, description="Extracted questions")
    questions_stored: int = Field(default=0, description="Number of questions stored")
    embeddings_count: int = Field(default=0, description="Number of embeddings generated")
    exam_metadata: Optional[Dict[str, Any]] = Field(None, description="Exam metadata")
    processing_time: float = Field(default=0.0, description="Processing time in seconds")
    error: Optional[str] = Field(None, description="Error message if failed")


class BatchProcessingResponse(BaseModel):
    """Batch processing response"""
    total_files: int = Field(..., description="Total number of files processed")
    successful_files: int = Field(..., description="Number of successfully processed files")
    failed_files: int = Field(..., description="Number of failed files")
    total_questions: int = Field(..., description="Total questions extracted")
    total_embeddings: int = Field(..., description="Total embeddings generated")
    processing_time: float = Field(..., description="Total processing time in seconds")
    file_results: List[ProcessingResultResponse] = Field(default_factory=list, description="Individual file results")
    errors: List[Dict[str, str]] = Field(default_factory=list, description="Processing errors")


class BulkOperationResponse(BaseModel):
    """Bulk operation response"""
    success: bool = Field(..., description="Whether operation was successful")
    message: str = Field(..., description="Response message")
    affected_count: int = Field(..., description="Number of questions affected")
    failed_ids: List[int] = Field(default_factory=list, description="IDs that failed to process")
    errors: List[str] = Field(default_factory=list, description="Error messages")


class SaveApprovedResponse(BaseModel):
    """Save approved questions response"""
    success: bool = Field(..., description="Whether operation was successful")
    message: str = Field(..., description="Response message")
    saved_count: int = Field(..., description="Number of questions saved")
    cleared_count: int = Field(default=0, description="Number of extracted questions cleared")
    failed_count: int = Field(default=0, description="Number of questions that failed to save")
    errors: List[str] = Field(default_factory=list, description="Error messages")


class ExportResponse(BaseModel):
    """Export response"""
    success: bool = Field(..., description="Whether export was successful")
    message: str = Field(..., description="Response message")
    filename: str = Field(..., description="Generated filename")
    download_url: str = Field(..., description="Download URL")
    record_count: int = Field(..., description="Number of records exported")
    file_size: int = Field(..., description="File size in bytes")
    format: str = Field(..., description="Export format")


class SearchResponse(BaseModel):
    """Search response"""
    results: List[QuestionResponse] = Field(..., description="Search results")
    total_results: int = Field(..., description="Total number of matching results")
    query: str = Field(..., description="Original search query")
    semantic_search_used: bool = Field(..., description="Whether semantic search was used")
    search_time: float = Field(..., description="Search execution time in seconds")
    filters_applied: Dict[str, Any] = Field(default_factory=dict, description="Applied filters")


class ErrorResponse(BaseModel):
    """Error response"""
    error: bool = Field(default=True, description="Indicates this is an error response")
    message: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    error_code: Optional[str] = Field(None, description="Error code")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ValidationErrorResponse(BaseModel):
    """Validation error response"""
    error: bool = Field(default=True, description="Indicates this is an error response")
    message: str = Field(default="Validation error", description="Error message")
    errors: List[Dict[str, Any]] = Field(..., description="Detailed validation errors")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class StatusResponse(BaseModel):
    """Generic status response"""
    success: bool = Field(..., description="Whether operation was successful")
    message: str = Field(..., description="Status message")
    data: Optional[Dict[str, Any]] = Field(None, description="Additional data")


class WebSocketResponse(BaseModel):
    """WebSocket response message"""
    type: str = Field(..., description="Message type")
    data: Optional[Dict[str, Any]] = Field(None, description="Message data")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Message timestamp")
    client_id: Optional[str] = Field(None, description="Client identifier")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class StatsResponse(BaseModel):
    """Statistics response"""
    total_extracted_questions: int = Field(..., description="Total extracted questions")
    total_approved_questions: int = Field(..., description="Total approved questions")
    total_permanent_questions: int = Field(..., description="Total permanent questions")
    questions_by_status: Dict[str, int] = Field(..., description="Question count by status")
    questions_by_year: Dict[str, int] = Field(..., description="Question count by year")
    questions_by_level: Dict[str, int] = Field(..., description="Question count by level")
    questions_by_type: Dict[str, int] = Field(..., description="Question count by type")
    recent_processing_activity: List[Dict[str, Any]] = Field(..., description="Recent processing activity")