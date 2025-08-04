import os
import uuid
import json
import asyncio
import logging
import tempfile
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any, Union
from io import StringIO, BytesIO

from fastapi import (
    APIRouter, Depends, HTTPException, UploadFile, File, Query, Body, Form,
    BackgroundTasks, WebSocket, WebSocketDisconnect, status
)
from fastapi.responses import StreamingResponse, FileResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_, text
from sqlalchemy.orm import selectinload

# Internal imports
from database.session import get_db
from database.models import ExtractedQuestion, Question, QuestionEmbedding
from services.pdf_processor import PDFQuestionProcessor, websocket_tracker
from api.schemas.requests import (
    UploadRequest, QuestionUpdateRequest, BulkOperationRequest,
    SaveApprovedRequest, ExportRequest, SearchRequest,
    QuestionStatusEnum, BulkOperationEnum, ExportFormatEnum
)
from api.schemas.responses import (
    UploadResponse, QuestionsListResponse, QuestionResponse,
    ProcessingProgressResponse, ProcessingResultResponse,
    BulkOperationResponse, SaveApprovedResponse, ExportResponse,
    SearchResponse, StatusResponse, ErrorResponse, StatsResponse
)
from config import Config

logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# Global processor instance for non-WebSocket operations
global_processor: Optional[PDFQuestionProcessor] = None


async def get_processor() -> PDFQuestionProcessor:
    """Get or create a global PDF processor instance"""
    global global_processor
    if global_processor is None:
        global_processor = PDFQuestionProcessor()
    return global_processor


# Utility functions

def model_to_dict(model_instance) -> Dict[str, Any]:
    """Convert SQLAlchemy model instance to dictionary"""
    return {
        column.name: getattr(model_instance, column.name)
        for column in model_instance.__table__.columns
    }


def question_to_response(question: Union[ExtractedQuestion, Question]) -> QuestionResponse:
    """Convert question model to response format"""
    return QuestionResponse(
        id=question.id,
        question_number=question.question_number,
        marks=question.marks,
        year=question.year,
        level=question.level,
        topics=question.topics or [],
        question_type=question.question_type,
        question_text=question.question_text,
        source_pdf=question.source_pdf,
        status=getattr(question, 'status', 'approved'),
        modified=getattr(question, 'modified', False),
        extraction_date=getattr(question, 'extraction_date', None),
        created_at=getattr(question, 'created_at', None),
        updated_at=getattr(question, 'updated_at', None),
        extra_metadata=getattr(question, 'extra_metadata', None) or getattr(question, 'metadata', None)
    )


async def export_to_csv(questions: List[QuestionResponse], include_metadata: bool = True) -> StringIO:
    """Export questions to CSV format"""
    import csv
    
    output = StringIO()
    
    # Define CSV headers
    headers = [
        'id', 'question_number', 'marks', 'year', 'level', 'topics',
        'question_type', 'question_text', 'source_pdf', 'status'
    ]
    
    if include_metadata:
        headers.extend(['extraction_date', 'created_at', 'updated_at', 'modified', 'extra_metadata'])
    
    writer = csv.DictWriter(output, fieldnames=headers)
    writer.writeheader()
    
    for question in questions:
        row = {
            'id': question.id,
            'question_number': question.question_number or '',
            'marks': question.marks or '',
            'year': question.year or '',
            'level': question.level or '',
            'topics': '|'.join(question.topics) if question.topics else '',
            'question_type': question.question_type or '',
            'question_text': question.question_text,
            'source_pdf': question.source_pdf,
            'status': question.status
        }
        
        if include_metadata:
            row.update({
                'extraction_date': question.extraction_date.isoformat() if question.extraction_date else '',
                'created_at': question.created_at.isoformat() if question.created_at else '',
                'updated_at': question.updated_at.isoformat() if question.updated_at else '',
                'modified': question.modified,
                'extra_metadata': json.dumps(question.extra_metadata) if question.extra_metadata else ''
            })
        
        writer.writerow(row)
    
    output.seek(0)
    return output


# API Endpoints

@router.post("/upload", response_model=UploadResponse)
async def upload_files(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    store_to_db: bool = Form(default=True),
    generate_embeddings: bool = Form(default=True),
    max_concurrent: int = Form(default=2),
    db: AsyncSession = Depends(get_db)
):
    """
    Upload PDF files or folders for processing
    
    - **files**: List of PDF files to upload
    - **store_to_db**: Whether to store extracted questions to database
    - **generate_embeddings**: Whether to generate vector embeddings
    - **max_concurrent**: Maximum concurrent file processing
    """
    try:
        processor = await get_processor()
        uploaded_files = []
        
        # Create upload directory if it doesn't exist
        upload_dir = Path(Config.UPLOAD_DIR)
        upload_dir.mkdir(exist_ok=True)
        
        # Save uploaded files
        for file in files:
            if not file.filename.lower().endswith('.pdf'):
                continue
            
            # Generate unique filename
            file_id = str(uuid.uuid4())
            filename = f"{file_id}_{file.filename}"
            file_path = upload_dir / filename
            
            # Save file
            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            
            uploaded_files.append(str(file_path))
        
        if not uploaded_files:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No valid PDF files provided"
            )
        
        # Generate processing ID
        processing_id = str(uuid.uuid4())
        
        # Start background processing
        if len(uploaded_files) == 1:
            background_tasks.add_task(
                processor.process_single_pdf,
                uploaded_files[0],
                store_to_db,
                generate_embeddings
            )
        else:
            # For multiple files, process as batch
            background_tasks.add_task(
                processor.process_pdf_folder,
                upload_dir,
                False,  # not recursive since files are in upload dir
                max_concurrent,
                store_to_db,
                generate_embeddings
            )
        
        return UploadResponse(
            success=True,
            message=f"Successfully uploaded {len(uploaded_files)} files for processing",
            processing_id=processing_id,
            files_uploaded=uploaded_files,
            total_files=len(uploaded_files)
        )
        
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Upload failed: {str(e)}"
        )


@router.get("/process/{processing_id}", response_model=Union[ProcessingProgressResponse, ProcessingResultResponse])
async def get_processing_status(
    processing_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Get processing status for a specific processing job
    
    - **processing_id**: ID of the processing job
    """
    # For this implementation, we'll return a generic response
    # In a production system, you'd want to track processing jobs in a database
    return ProcessingProgressResponse(
        file_path=f"processing_{processing_id}",
        status="processing",
        current_step="Processing files...",
        total_steps=5,
        completed_steps=2,
        progress_percentage=40.0,
        start_time=datetime.utcnow(),
        questions_extracted=0,
        questions_stored=0,
        embeddings_generated=0,
        is_complete=False
    )


@router.get("/questions", response_model=QuestionsListResponse)
async def get_questions(
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(20, ge=1, le=100, description="Items per page"),
    status_filter: Optional[QuestionStatusEnum] = Query(None, description="Filter by status"),
    year_filter: Optional[str] = Query(None, description="Filter by year"),
    level_filter: Optional[str] = Query(None, description="Filter by level"),
    question_type_filter: Optional[str] = Query(None, description="Filter by question type"),
    source_pdf_filter: Optional[str] = Query(None, description="Filter by source PDF"),
    search: Optional[str] = Query(None, description="Search in question text"),
    table: str = Query("extracted", description="Table to query: 'extracted' or 'permanent'"),
    db: AsyncSession = Depends(get_db)
):
    """
    Get paginated list of questions with optional filtering
    
    - **page**: Page number (starts from 1)
    - **per_page**: Number of items per page
    - **status_filter**: Filter by question status
    - **year_filter**: Filter by exam year
    - **level_filter**: Filter by education level
    - **question_type_filter**: Filter by question type
    - **source_pdf_filter**: Filter by source PDF filename
    - **search**: Search term for question text
    - **table**: Which table to query ('extracted' or 'permanent')
    """
    try:
        # Choose table
        model = ExtractedQuestion if table == "extracted" else Question
        
        # Build query
        query = select(model)
        count_query = select(func.count(model.id))
        
        # Apply filters
        filters = []
        
        if status_filter and hasattr(model, 'status'):
            filters.append(model.status == status_filter.value)
        
        if year_filter:
            filters.append(model.year == year_filter)
        
        if level_filter:
            filters.append(model.level == level_filter)
        
        if question_type_filter:
            filters.append(model.question_type == question_type_filter)
        
        if source_pdf_filter:
            filters.append(model.source_pdf.ilike(f"%{source_pdf_filter}%"))
        
        if search:
            filters.append(model.question_text.ilike(f"%{search}%"))
        
        if filters:
            query = query.where(and_(*filters))
            count_query = count_query.where(and_(*filters))
        
        # Get total count
        result = await db.execute(count_query)
        total = result.scalar() or 0
        
        # Calculate pagination
        offset = (page - 1) * per_page
        total_pages = (total + per_page - 1) // per_page
        has_next = page < total_pages
        has_prev = page > 1
        
        # Apply pagination and ordering
        query = query.offset(offset).limit(per_page).order_by(model.id.desc())
        
        # Execute query
        result = await db.execute(query)
        questions = result.scalars().all()
        
        # Convert to response format
        question_responses = [question_to_response(q) for q in questions]
        
        return QuestionsListResponse(
            questions=question_responses,
            total=total,
            page=page,
            per_page=per_page,
            total_pages=total_pages,
            has_next=has_next,
            has_prev=has_prev
        )
        
    except Exception as e:
        logger.error(f"Error getting questions: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve questions: {str(e)}"
        )


@router.put("/questions/{question_id}", response_model=QuestionResponse)
async def update_question(
    question_id: int,
    update_data: QuestionUpdateRequest,
    table: str = Query("extracted", description="Table to update: 'extracted' or 'permanent'"),
    db: AsyncSession = Depends(get_db)
):
    """
    Update a single question
    
    - **question_id**: ID of the question to update
    - **update_data**: Updated question data
    - **table**: Which table to update ('extracted' or 'permanent')
    """
    try:
        # Choose model
        model = ExtractedQuestion if table == "extracted" else Question
        
        # Get existing question
        query = select(model).where(model.id == question_id)
        result = await db.execute(query)
        question = result.scalar_one_or_none()
        
        if not question:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Question with ID {question_id} not found"
            )
        
        # Update fields
        update_dict = update_data.dict(exclude_unset=True)
        
        for field, value in update_dict.items():
            if hasattr(question, field):
                setattr(question, field, value)
        
        # Mark as modified if it's an extracted question
        if hasattr(question, 'modified'):
            question.modified = True
        
        # Update timestamp
        if hasattr(question, 'updated_at'):
            question.updated_at = datetime.utcnow()
        
        await db.commit()
        await db.refresh(question)
        
        return question_to_response(question)
        
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        logger.error(f"Error updating question {question_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update question: {str(e)}"
        )


@router.post("/questions/bulk", response_model=BulkOperationResponse)
async def bulk_operations(
    bulk_request: BulkOperationRequest,
    table: str = Query("extracted", description="Table to operate on: 'extracted' or 'permanent'"),
    db: AsyncSession = Depends(get_db)
):
    """
    Perform bulk operations on questions
    
    - **bulk_request**: Bulk operation request data
    - **table**: Which table to operate on ('extracted' or 'permanent')
    """
    try:
        model = ExtractedQuestion if table == "extracted" else Question
        affected_count = 0
        failed_ids = []
        errors = []
        
        # Get questions
        query = select(model).where(model.id.in_(bulk_request.question_ids))
        result = await db.execute(query)
        questions = result.scalars().all()
        
        if not questions:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No questions found with provided IDs"
            )
        
        # Perform operation
        for question in questions:
            try:
                if bulk_request.operation == BulkOperationEnum.DELETE:
                    await db.delete(question)
                    affected_count += 1
                
                elif bulk_request.operation == BulkOperationEnum.UPDATE_STATUS:
                    if hasattr(question, 'status'):
                        question.status = bulk_request.new_status.value
                        if hasattr(question, 'modified'):
                            question.modified = True
                        affected_count += 1
                
                elif bulk_request.operation == BulkOperationEnum.APPROVE:
                    if hasattr(question, 'status'):
                        question.status = QuestionStatusEnum.APPROVED.value
                        if hasattr(question, 'modified'):
                            question.modified = True
                        affected_count += 1
                
                elif bulk_request.operation == BulkOperationEnum.REJECT:
                    if hasattr(question, 'status'):
                        question.status = QuestionStatusEnum.REJECTED.value
                        if hasattr(question, 'modified'):
                            question.modified = True
                        affected_count += 1
                
            except Exception as e:
                failed_ids.append(question.id)
                errors.append(f"Question {question.id}: {str(e)}")
        
        await db.commit()
        
        return BulkOperationResponse(
            success=True,
            message=f"Bulk operation completed. {affected_count} questions affected.",
            affected_count=affected_count,
            failed_ids=failed_ids,
            errors=errors
        )
        
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        logger.error(f"Bulk operation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Bulk operation failed: {str(e)}"
        )


@router.post("/questions/save", response_model=SaveApprovedResponse)
async def save_approved_questions(
    save_request: SaveApprovedRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Save approved questions from extracted_questions to permanent questions table
    
    - **save_request**: Save operation parameters
    """
    try:
        saved_count = 0
        cleared_count = 0
        failed_count = 0
        errors = []
        
        # Build query for approved questions
        query = select(ExtractedQuestion).where(
            ExtractedQuestion.status == QuestionStatusEnum.APPROVED.value
        )
        
        if save_request.question_ids:
            query = query.where(ExtractedQuestion.id.in_(save_request.question_ids))
        
        result = await db.execute(query)
        approved_questions = result.scalars().all()
        
        if not approved_questions:
            return SaveApprovedResponse(
                success=True,
                message="No approved questions found to save",
                saved_count=0,
                cleared_count=0,
                failed_count=0,
                errors=[]
            )
        
        # Save each approved question to permanent table
        for extracted_q in approved_questions:
            try:
                # Create permanent question
                permanent_q = Question(
                    question_number=extracted_q.question_number,
                    marks=extracted_q.marks,
                    year=extracted_q.year,
                    level=extracted_q.level,
                    topics=extracted_q.topics,
                    question_type=extracted_q.question_type,
                    question_text=extracted_q.question_text,
                    source_pdf=extracted_q.source_pdf,
                    extra_metadata=extracted_q.extra_metadata
                )
                
                db.add(permanent_q)
                await db.flush()  # Get the ID
                saved_count += 1
                
                # Clear from extracted table if requested
                if save_request.clear_extracted:
                    await db.delete(extracted_q)
                    cleared_count += 1
                
            except Exception as e:
                failed_count += 1
                errors.append(f"Question {extracted_q.id}: {str(e)}")
        
        await db.commit()
        
        return SaveApprovedResponse(
            success=True,
            message=f"Successfully saved {saved_count} approved questions",
            saved_count=saved_count,
            cleared_count=cleared_count,
            failed_count=failed_count,
            errors=errors
        )
        
    except Exception as e:
        await db.rollback()
        logger.error(f"Save approved questions error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save approved questions: {str(e)}"
        )


@router.get("/export", response_model=ExportResponse)
async def export_questions(
    export_request: ExportRequest = Depends(),
    db: AsyncSession = Depends(get_db)
):
    """
    Export questions to CSV, JSON, or Excel format
    
    - **export_request**: Export parameters and filters
    """
    try:
        # Choose table
        model = ExtractedQuestion if export_request.status_filter else Question
        
        # Build query
        query = select(model)
        filters = []
        
        # Apply filters
        if export_request.status_filter and hasattr(model, 'status'):
            filters.append(model.status == export_request.status_filter.value)
        
        if export_request.year_filter:
            filters.append(model.year == export_request.year_filter)
        
        if export_request.level_filter:
            filters.append(model.level == export_request.level_filter)
        
        if export_request.question_type_filter:
            filters.append(model.question_type == export_request.question_type_filter.value)
        
        if export_request.source_pdf_filter:
            filters.append(model.source_pdf.ilike(f"%{export_request.source_pdf_filter}%"))
        
        if export_request.topics_filter:
            # AND logic for topics
            for topic in export_request.topics_filter:
                filters.append(model.topics.contains([topic]))
        
        if export_request.question_ids:
            filters.append(model.id.in_(export_request.question_ids))
        
        if filters:
            query = query.where(and_(*filters))
        
        # Execute query
        result = await db.execute(query)
        questions = result.scalars().all()
        
        # Convert to response format
        question_responses = [question_to_response(q) for q in questions]
        
        # Generate filename
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"questions_export_{timestamp}.{export_request.format.value}"
        
        # Create export directory
        export_dir = Path(Config.CACHE_DIR) / "exports"
        export_dir.mkdir(exist_ok=True)
        file_path = export_dir / filename
        
        if export_request.format == ExportFormatEnum.CSV:
            # Export to CSV
            csv_output = await export_to_csv(question_responses, export_request.include_metadata)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(csv_output.getvalue())
        
        elif export_request.format == ExportFormatEnum.JSON:
            # Export to JSON
            export_data = [q.dict() for q in question_responses]
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, default=str)
        
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Export format {export_request.format} not yet implemented"
            )
        
        # Get file size
        file_size = file_path.stat().st_size
        
        return ExportResponse(
            success=True,
            message=f"Successfully exported {len(question_responses)} questions",
            filename=filename,
            download_url=f"/api/download/{filename}",
            record_count=len(question_responses),
            file_size=file_size,
            format=export_request.format.value
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Export error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Export failed: {str(e)}"
        )


@router.get("/download/{filename}")
async def download_file(filename: str):
    """
    Download exported file
    
    - **filename**: Name of the file to download
    """
    try:
        export_dir = Path(Config.CACHE_DIR) / "exports"
        file_path = export_dir / filename
        
        if not file_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="File not found"
            )
        
        return FileResponse(
            path=file_path,
            filename=filename,
            media_type='application/octet-stream'
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Download error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Download failed: {str(e)}"
        )


@router.get("/stats", response_model=StatsResponse)
async def get_statistics(db: AsyncSession = Depends(get_db)):
    """
    Get system statistics and metrics
    """
    try:
        # Get extracted questions stats
        extracted_count_query = select(func.count(ExtractedQuestion.id))
        result = await db.execute(extracted_count_query)
        total_extracted = result.scalar() or 0
        
        # Get permanent questions stats
        permanent_count_query = select(func.count(Question.id))
        result = await db.execute(permanent_count_query)
        total_permanent = result.scalar() or 0
        
        # Get approved count
        approved_count_query = select(func.count(ExtractedQuestion.id)).where(
            ExtractedQuestion.status == QuestionStatusEnum.APPROVED.value
        )
        result = await db.execute(approved_count_query)
        total_approved = result.scalar() or 0
        
        # Questions by status
        status_query = select(
            ExtractedQuestion.status,
            func.count(ExtractedQuestion.id)
        ).group_by(ExtractedQuestion.status)
        result = await db.execute(status_query)
        questions_by_status = dict(result.all())
        
        # Questions by year (from both tables)
        year_query = select(
            Question.year,
            func.count(Question.id)
        ).where(Question.year.isnot(None)).group_by(Question.year)
        result = await db.execute(year_query)
        questions_by_year = dict(result.all())
        
        # Questions by level
        level_query = select(
            Question.level,
            func.count(Question.id)
        ).where(Question.level.isnot(None)).group_by(Question.level)
        result = await db.execute(level_query)
        questions_by_level = dict(result.all())
        
        # Questions by type
        type_query = select(
            Question.question_type,
            func.count(Question.id)
        ).where(Question.question_type.isnot(None)).group_by(Question.question_type)
        result = await db.execute(type_query)
        questions_by_type = dict(result.all())
        
        return StatsResponse(
            total_extracted_questions=total_extracted,
            total_approved_questions=total_approved,
            total_permanent_questions=total_permanent,
            questions_by_status=questions_by_status,
            questions_by_year=questions_by_year,
            questions_by_level=questions_by_level,
            questions_by_type=questions_by_type,
            recent_processing_activity=[]  # Placeholder
        )
        
    except Exception as e:
        logger.error(f"Stats error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get statistics: {str(e)}"
        )


# WebSocket endpoint for real-time processing updates
@router.websocket("/ws/processing")
async def websocket_processing_updates(websocket: WebSocket):
    """
    WebSocket endpoint for real-time processing updates
    
    Provides live updates on PDF processing progress
    """
    await websocket.accept()
    client_id = str(uuid.uuid4())
    
    try:
        # Add client to tracker
        await websocket_tracker.add_client(client_id, websocket)
        
        # Send welcome message
        welcome_message = {
            "type": "connection_established",
            "data": {
                "client_id": client_id,
                "message": "Connected to processing updates"
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        await websocket.send_text(json.dumps(welcome_message))
        
        # Keep connection alive and handle incoming messages
        while True:
            try:
                # Wait for messages from client
                data = await websocket.receive_text()
                message = json.loads(data)
                
                # Handle different message types
                if message.get("type") == "ping":
                    # Respond to ping
                    pong_message = {
                        "type": "pong",
                        "timestamp": datetime.utcnow().isoformat()
                    }
                    await websocket.send_text(json.dumps(pong_message))
                
                elif message.get("type") == "start_processing":
                    # Start processing request
                    processor = websocket_tracker.get_processor(client_id)
                    if processor:
                        # Handle processing start
                        response_message = {
                            "type": "processing_started",
                            "data": {
                                "message": "Processing request received"
                            },
                            "timestamp": datetime.utcnow().isoformat()
                        }
                        await websocket.send_text(json.dumps(response_message))
                
            except WebSocketDisconnect:
                break
            except json.JSONDecodeError:
                # Send error message for invalid JSON
                error_message = {
                    "type": "error",
                    "data": {
                        "message": "Invalid JSON format"
                    },
                    "timestamp": datetime.utcnow().isoformat()
                }
                await websocket.send_text(json.dumps(error_message))
    
    except WebSocketDisconnect:
        logger.info(f"WebSocket client {client_id} disconnected")
    except Exception as e:
        logger.error(f"WebSocket error for client {client_id}: {str(e)}")
    finally:
        # Clean up client
        await websocket_tracker.remove_client(client_id)