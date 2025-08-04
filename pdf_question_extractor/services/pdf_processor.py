import os
import asyncio
import logging
from typing import List, Dict, Any, Optional, Callable, Union
from pathlib import Path
from enum import Enum
from datetime import datetime
import json
from dataclasses import dataclass, asdict
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import AsyncSession
from database.session import AsyncSessionLocal
from database.models import ExtractedQuestion, Question, QuestionEmbedding
from services.ocr_service import MistralOCRService
from services.llm_service import GeminiLLMService, ExamPaper, Question as LLMQuestion
from services.embedding_service import GeminiEmbeddingService
from config import Config

logger = logging.getLogger(__name__)

# Processing status enum
class ProcessingStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    OCR_COMPLETE = "ocr_complete"
    LLM_COMPLETE = "llm_complete"
    EMBEDDING_COMPLETE = "embedding_complete"
    STORED = "stored"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

# Progress tracking dataclass
@dataclass
class ProcessingProgress:
    file_path: str
    status: ProcessingStatus
    current_step: str
    total_steps: int
    completed_steps: int
    start_time: datetime
    end_time: Optional[datetime] = None
    error_message: Optional[str] = None
    questions_extracted: int = 0
    questions_stored: int = 0
    embeddings_generated: int = 0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    @property
    def progress_percentage(self) -> float:
        """Calculate progress as percentage"""
        if self.total_steps == 0:
            return 0.0
        return (self.completed_steps / self.total_steps) * 100
    
    @property
    def is_complete(self) -> bool:
        """Check if processing is complete"""
        return self.status in [ProcessingStatus.COMPLETED, ProcessingStatus.FAILED, ProcessingStatus.CANCELLED]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        data['start_time'] = self.start_time.isoformat() if self.start_time else None
        data['end_time'] = self.end_time.isoformat() if self.end_time else None
        data['progress_percentage'] = self.progress_percentage
        data['is_complete'] = self.is_complete
        return data

# Batch processing results
@dataclass
class BatchProcessingResult:
    total_files: int
    successful_files: int
    failed_files: int
    total_questions: int
    total_embeddings: int
    processing_time: float
    file_results: List[Dict[str, Any]]
    errors: List[Dict[str, Any]]

class PDFQuestionProcessor:
    """Main orchestrator for PDF question extraction pipeline"""
    
    def __init__(
        self,
        mistral_api_key: Optional[str] = None,
        google_api_key: Optional[str] = None,
        progress_callback: Optional[Callable[[ProcessingProgress], None]] = None
    ):
        """
        Initialize PDF Question Processor with all required services
        
        Args:
            mistral_api_key: Mistral API key for OCR (defaults to config)
            google_api_key: Google API key for LLM and embeddings (defaults to config)
            progress_callback: Optional callback for progress updates
        """
        # Initialize API keys
        self.mistral_api_key = mistral_api_key or Config.MISTRAL_API_KEY
        self.google_api_key = google_api_key or Config.GOOGLE_API_KEY
        
        if not self.mistral_api_key:
            raise ValueError("Mistral API key is required")
        if not self.google_api_key:
            raise ValueError("Google API key is required")
        
        # Initialize services
        self.ocr_service = MistralOCRService(self.mistral_api_key)
        self.llm_service = GeminiLLMService(self.google_api_key)
        self.embedding_service = GeminiEmbeddingService(self.google_api_key)
        
        # Note: VectorOperations may need adjustment - using direct database operations instead
        # self.vector_ops = VectorOperations()
        
        # Progress tracking
        self.progress_callback = progress_callback
        self._active_processes: Dict[str, ProcessingProgress] = {}
        self._process_lock = asyncio.Lock()
        
        # Configuration
        self.max_file_size = Config.MAX_UPLOAD_SIZE
        self.supported_extensions = {'.pdf'}
        
        logger.info("PDFQuestionProcessor initialized")
    
    def set_progress_callback(self, callback: Callable[[ProcessingProgress], None]):
        """Set or update the progress callback"""
        self.progress_callback = callback
    
    async def _update_progress(
        self, 
        file_path: str, 
        status: ProcessingStatus = None,
        current_step: str = None,
        increment_steps: int = 0,
        error_message: str = None,
        **kwargs
    ):
        """Update processing progress and notify callback"""
        async with self._process_lock:
            if file_path not in self._active_processes:
                return
            
            progress = self._active_processes[file_path]
            
            if status:
                progress.status = status
            if current_step:
                progress.current_step = current_step
            if increment_steps > 0:
                progress.completed_steps += increment_steps
            if error_message:
                progress.error_message = error_message
                progress.status = ProcessingStatus.FAILED
                progress.end_time = datetime.utcnow()
            
            # Update additional fields
            for key, value in kwargs.items():
                if hasattr(progress, key):
                    setattr(progress, key, value)
            
            # Mark as complete if status indicates completion
            if progress.status in [ProcessingStatus.COMPLETED, ProcessingStatus.FAILED, ProcessingStatus.CANCELLED]:
                progress.end_time = datetime.utcnow()
            
            # Notify callback if provided
            if self.progress_callback:
                try:
                    self.progress_callback(progress)
                except Exception as e:
                    logger.error(f"Error in progress callback: {e}")
    
    async def process_single_pdf(
        self, 
        pdf_path: Union[str, Path],
        store_to_db: bool = True,
        generate_embeddings: bool = True
    ) -> Dict[str, Any]:
        """
        Process a single PDF file through the complete pipeline
        
        Args:
            pdf_path: Path to PDF file
            store_to_db: Whether to store results in database
            generate_embeddings: Whether to generate embeddings
            
        Returns:
            Dictionary with processing results
        """
        pdf_path = Path(pdf_path)
        file_key = str(pdf_path)
        
        # Validate file
        validation_result = await self._validate_pdf_file(pdf_path)
        if not validation_result['valid']:
            return {
                'success': False,
                'file_path': str(pdf_path),
                'error': validation_result['error'],
                'questions': [],
                'embeddings_count': 0
            }
        
        # Initialize progress tracking
        total_steps = 3  # OCR, LLM, Store
        if generate_embeddings:
            total_steps += 1
        
        progress = ProcessingProgress(
            file_path=file_key,
            status=ProcessingStatus.PENDING,
            current_step="Initializing",
            total_steps=total_steps,
            completed_steps=0,
            start_time=datetime.utcnow()
        )
        
        async with self._process_lock:
            self._active_processes[file_key] = progress
        
        try:
            # Step 1: OCR Processing
            await self._update_progress(
                file_key, 
                status=ProcessingStatus.PROCESSING,
                current_step="Extracting text with OCR"
            )
            
            ocr_text = await self.ocr_service.process_pdf(str(pdf_path))
            
            await self._update_progress(
                file_key,
                status=ProcessingStatus.OCR_COMPLETE,
                current_step="OCR completed",
                increment_steps=1
            )
            
            # Step 2: LLM Question Extraction
            await self._update_progress(
                file_key, 
                current_step="Extracting questions with LLM"
            )
            
            exam_paper = await self.llm_service.extract_questions(
                ocr_text, 
                pdf_path.name
            )
            
            await self._update_progress(
                file_key,
                status=ProcessingStatus.LLM_COMPLETE,
                current_step="Question extraction completed",
                increment_steps=1,
                questions_extracted=len(exam_paper.questions)
            )
            
            # Step 3: Store to Database (if requested)
            stored_questions = []
            if store_to_db:
                await self._update_progress(
                    file_key,
                    current_step="Storing questions to database"
                )
                
                stored_questions = await self._store_questions_to_db(exam_paper)
                
                await self._update_progress(
                    file_key,
                    status=ProcessingStatus.STORED,
                    current_step="Questions stored to database",
                    increment_steps=1,
                    questions_stored=len(stored_questions)
                )
            
            # Step 4: Generate Embeddings (if requested)
            embeddings_count = 0
            if generate_embeddings and stored_questions:
                await self._update_progress(
                    file_key,
                    current_step="Generating embeddings"
                )
                
                embeddings_count = await self._generate_embeddings_for_questions(
                    stored_questions, file_key
                )
                
                await self._update_progress(
                    file_key,
                    status=ProcessingStatus.EMBEDDING_COMPLETE,
                    current_step="Embeddings generated",
                    increment_steps=1,
                    embeddings_generated=embeddings_count
                )
            
            # Mark as completed
            await self._update_progress(
                file_key,
                status=ProcessingStatus.COMPLETED,
                current_step="Processing completed"
            )
            
            return {
                'success': True,
                'file_path': str(pdf_path),
                'questions': [self._llm_question_to_dict(q) for q in exam_paper.questions],
                'questions_stored': len(stored_questions),
                'embeddings_count': embeddings_count,
                'exam_metadata': {
                    'year': exam_paper.year,
                    'level': exam_paper.level,
                    'subject': exam_paper.subject,
                    'total_marks': exam_paper.total_marks,
                    'duration': exam_paper.duration
                },
                'processing_time': (datetime.utcnow() - progress.start_time).total_seconds()
            }
            
        except Exception as e:
            logger.error(f"Error processing {pdf_path}: {str(e)}")
            await self._update_progress(
                file_key,
                error_message=str(e)
            )
            
            return {
                'success': False,
                'file_path': str(pdf_path),
                'error': str(e),
                'questions': [],
                'embeddings_count': 0
            }
        
        finally:
            # Clean up progress tracking
            async with self._process_lock:
                if file_key in self._active_processes:
                    del self._active_processes[file_key]
    
    async def process_pdf_folder(
        self,
        folder_path: Union[str, Path],
        recursive: bool = True,
        max_concurrent: int = 3,
        store_to_db: bool = True,
        generate_embeddings: bool = True
    ) -> BatchProcessingResult:
        """
        Process all PDF files in a folder
        
        Args:
            folder_path: Path to folder containing PDFs
            recursive: Whether to search subdirectories
            max_concurrent: Maximum number of files to process concurrently
            store_to_db: Whether to store results in database
            generate_embeddings: Whether to generate embeddings
            
        Returns:
            BatchProcessingResult with summary statistics
        """
        folder_path = Path(folder_path)
        start_time = datetime.utcnow()
        
        if not folder_path.exists() or not folder_path.is_dir():
            raise ValueError(f"Invalid folder path: {folder_path}")
        
        # Find all PDF files
        pdf_files = self._find_pdf_files(folder_path, recursive)
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {folder_path}")
            return BatchProcessingResult(
                total_files=0,
                successful_files=0,
                failed_files=0,
                total_questions=0,
                total_embeddings=0,
                processing_time=0.0,
                file_results=[],
                errors=[]
            )
        
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        # Process files with concurrency control
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_with_semaphore(pdf_file):
            async with semaphore:
                return await self.process_single_pdf(
                    pdf_file, store_to_db, generate_embeddings
                )
        
        # Execute all processing tasks
        tasks = [process_with_semaphore(pdf_file) for pdf_file in pdf_files]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Analyze results
        successful_files = 0
        failed_files = 0
        total_questions = 0
        total_embeddings = 0
        file_results = []
        errors = []
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                failed_files += 1
                errors.append({
                    'file_path': str(pdf_files[i]),
                    'error': str(result)
                })
            elif result['success']:
                successful_files += 1
                total_questions += len(result['questions'])
                total_embeddings += result['embeddings_count']
                file_results.append(result)
            else:
                failed_files += 1
                errors.append({
                    'file_path': result['file_path'],
                    'error': result.get('error', 'Unknown error')
                })
        
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        logger.info(f"Batch processing completed: {successful_files}/{len(pdf_files)} files successful")
        
        return BatchProcessingResult(
            total_files=len(pdf_files),
            successful_files=successful_files,
            failed_files=failed_files,
            total_questions=total_questions,
            total_embeddings=total_embeddings,
            processing_time=processing_time,
            file_results=file_results,
            errors=errors
        )
    
    async def get_processing_status(self, file_path: str) -> Optional[ProcessingProgress]:
        """Get current processing status for a file"""
        async with self._process_lock:
            return self._active_processes.get(file_path)
    
    async def get_all_processing_status(self) -> Dict[str, ProcessingProgress]:
        """Get processing status for all active processes"""
        async with self._process_lock:
            return self._active_processes.copy()
    
    async def cancel_processing(self, file_path: str) -> bool:
        """Cancel processing for a specific file"""
        async with self._process_lock:
            if file_path in self._active_processes:
                await self._update_progress(
                    file_path,
                    status=ProcessingStatus.CANCELLED,
                    current_step="Processing cancelled"
                )
                return True
            return False
    
    # Private helper methods
    
    async def _validate_pdf_file(self, pdf_path: Path) -> Dict[str, Any]:
        """Validate PDF file before processing"""
        if not pdf_path.exists():
            return {'valid': False, 'error': 'File does not exist'}
        
        if pdf_path.suffix.lower() not in self.supported_extensions:
            return {'valid': False, 'error': f'Unsupported file type: {pdf_path.suffix}'}
        
        file_size = pdf_path.stat().st_size
        if file_size > self.max_file_size:
            return {
                'valid': False, 
                'error': f'File too large: {file_size / 1024 / 1024:.2f}MB (max {self.max_file_size / 1024 / 1024:.2f}MB)'
            }
        
        if file_size == 0:
            return {'valid': False, 'error': 'File is empty'}
        
        return {'valid': True}
    
    def _find_pdf_files(self, folder_path: Path, recursive: bool = True) -> List[Path]:
        """Find all PDF files in a folder"""
        pdf_files = []
        
        if recursive:
            pdf_files = list(folder_path.rglob('*.pdf'))
        else:
            pdf_files = list(folder_path.glob('*.pdf'))
        
        # Filter out hidden files and ensure they're readable
        valid_files = []
        for pdf_file in pdf_files:
            if not pdf_file.name.startswith('.') and pdf_file.is_file():
                valid_files.append(pdf_file)
        
        return sorted(valid_files)
    
    async def _store_questions_to_db(self, exam_paper: ExamPaper) -> List[Dict[str, Any]]:
        """Store extracted questions to database"""
        stored_questions = []
        
        async with AsyncSessionLocal() as session:
            try:
                for question in exam_paper.questions:
                    # Create ExtractedQuestion for review
                    extracted_question = ExtractedQuestion(
                        question_number=question.question_number,
                        marks=question.marks,
                        year=exam_paper.year,
                        level=exam_paper.level,
                        topics=question.topics,
                        question_type=question.question_type.value,
                        question_text=question.question_text,
                        source_pdf=exam_paper.source_pdf,
                        status='pending',
                        modified=False,
                        extra_metadata={
                            'total_marks': exam_paper.total_marks,
                            'duration': exam_paper.duration,
                            'subject': exam_paper.subject
                        }
                    )
                    
                    session.add(extracted_question)
                    await session.flush()  # Get the ID
                    
                    stored_questions.append({
                        'id': extracted_question.id,
                        'question_number': extracted_question.question_number,
                        'marks': extracted_question.marks,
                        'question_text': extracted_question.question_text,
                        'topics': extracted_question.topics,
                        'question_type': extracted_question.question_type
                    })
                
                await session.commit()
                logger.info(f"Stored {len(stored_questions)} questions to database")
                
            except Exception as e:
                await session.rollback()
                logger.error(f"Error storing questions to database: {str(e)}")
                raise
        
        return stored_questions
    
    async def _generate_embeddings_for_questions(
        self, 
        stored_questions: List[Dict[str, Any]],
        file_key: str
    ) -> int:
        """Generate embeddings for stored questions"""
        if not stored_questions:
            return 0
        
        try:
            # Prepare question data for embedding generation
            questions_for_embedding = []
            for q in stored_questions:
                questions_for_embedding.append({
                    'id': q['id'],
                    'question_text': q['question_text'],
                    'question_type': q['question_type'],
                    'topics': q['topics'],
                    'marks': q['marks']
                })
            
            # Generate embeddings in batches
            embeddings = await self.embedding_service.generate_batch_embeddings(
                questions_for_embedding
            )
            
            # Store embeddings in database
            stored_count = 0
            async with AsyncSessionLocal() as session:
                for question_id, embedding_vector in embeddings.items():
                    try:
                        # Check if this is for an ExtractedQuestion or Question
                        # For now, we'll assume ExtractedQuestion since we just stored them
                        # In a production system, you might want to store embeddings 
                        # only after questions are approved and moved to the Question table
                        
                        new_embedding = QuestionEmbedding(
                            question_id=question_id,
                            embedding=embedding_vector,
                            model_name='gemini-embedding-001',
                            model_version='1.0'
                        )
                        
                        session.add(new_embedding)
                        stored_count += 1
                        
                    except Exception as e:
                        logger.error(f"Error storing embedding for question {question_id}: {e}")
                
                await session.commit()
            
            logger.info(f"Generated and stored {stored_count} embeddings")
            return stored_count
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            return 0
    
    def _llm_question_to_dict(self, question: LLMQuestion) -> Dict[str, Any]:
        """Convert LLM Question object to dictionary"""
        return {
            'question_number': question.question_number,
            'marks': question.marks,
            'question_text': question.question_text,
            'topics': question.topics,
            'question_type': question.question_type.value
        }
    
    # Context manager support
    
    async def __aenter__(self):
        """Async context manager entry"""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        # Cancel any remaining active processes
        async with self._process_lock:
            for file_path in list(self._active_processes.keys()):
                await self.cancel_processing(file_path)

# WebSocket Progress Tracker
class WebSocketProgressTracker:
    """WebSocket-based progress tracking for real-time updates"""
    
    def __init__(self):
        self.connections: Dict[str, Any] = {}  # WebSocket connections by client_id
        self.client_processors: Dict[str, PDFQuestionProcessor] = {}  # Processors by client_id
    
    async def add_client(self, client_id: str, websocket):
        """Add a WebSocket client"""
        self.connections[client_id] = websocket
        
        # Create processor with progress callback
        processor = PDFQuestionProcessor(
            progress_callback=lambda progress: asyncio.create_task(
                self._send_progress_update(client_id, progress)
            )
        )
        self.client_processors[client_id] = processor
        
        logger.info(f"Added WebSocket client: {client_id}")
    
    async def remove_client(self, client_id: str):
        """Remove a WebSocket client"""
        if client_id in self.connections:
            del self.connections[client_id]
        
        if client_id in self.client_processors:
            # Cancel any active processing
            processor = self.client_processors[client_id]
            active_processes = await processor.get_all_processing_status()
            for file_path in active_processes.keys():
                await processor.cancel_processing(file_path)
            
            del self.client_processors[client_id]
        
        logger.info(f"Removed WebSocket client: {client_id}")
    
    async def _send_progress_update(self, client_id: str, progress: ProcessingProgress):
        """Send progress update to specific client"""
        if client_id not in self.connections:
            return
        
        try:
            websocket = self.connections[client_id]
            message = {
                'type': 'progress_update',
                'data': progress.to_dict()
            }
            await websocket.send_text(json.dumps(message))
        except Exception as e:
            logger.error(f"Error sending progress update to {client_id}: {e}")
            # Remove client if connection is broken
            await self.remove_client(client_id)
    
    def get_processor(self, client_id: str) -> Optional[PDFQuestionProcessor]:
        """Get processor for a specific client"""
        return self.client_processors.get(client_id)

# Global WebSocket tracker instance
websocket_tracker = WebSocketProgressTracker()

# Convenience functions for common operations

async def process_pdf_file(
    pdf_path: Union[str, Path], 
    progress_callback: Optional[Callable] = None
) -> Dict[str, Any]:
    """Convenience function to process a single PDF file"""
    async with PDFQuestionProcessor(progress_callback=progress_callback) as processor:
        return await processor.process_single_pdf(pdf_path)

async def process_pdf_directory(
    directory_path: Union[str, Path],
    recursive: bool = True,
    max_concurrent: int = 3,
    progress_callback: Optional[Callable] = None
) -> BatchProcessingResult:
    """Convenience function to process a directory of PDF files"""
    async with PDFQuestionProcessor(progress_callback=progress_callback) as processor:
        return await processor.process_pdf_folder(
            directory_path, recursive, max_concurrent
        )

# Example usage and testing functions

async def example_usage():
    """Example usage of the PDF Question Processor"""
    
    def progress_callback(progress: ProcessingProgress):
        print(f"[{progress.file_path}] {progress.current_step} ({progress.progress_percentage:.1f}%)")
        if progress.is_complete:
            if progress.status == ProcessingStatus.COMPLETED:
                print(f" Completed: {progress.questions_extracted} questions, {progress.embeddings_generated} embeddings")
            else:
                print(f"L Failed: {progress.error_message}")
    
    # Process single file
    result = await process_pdf_file(
        "path/to/exam.pdf",
        progress_callback=progress_callback
    )
    
    if result['success']:
        print(f"Successfully processed {result['file_path']}")
        print(f"Extracted {len(result['questions'])} questions")
    else:
        print(f"Failed to process: {result['error']}")
    
    # Process directory
    batch_result = await process_pdf_directory(
        "path/to/exam_folder",
        recursive=True,
        max_concurrent=2,
        progress_callback=progress_callback
    )
    
    print(f"\nBatch processing results:")
    print(f"Total files: {batch_result.total_files}")
    print(f"Successful: {batch_result.successful_files}")
    print(f"Failed: {batch_result.failed_files}")
    print(f"Total questions: {batch_result.total_questions}")
    print(f"Total embeddings: {batch_result.total_embeddings}")
    print(f"Processing time: {batch_result.processing_time:.2f}s")

if __name__ == "__main__":
    # Run example usage
    asyncio.run(example_usage())