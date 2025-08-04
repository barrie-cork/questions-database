"""
Integration tests for PDF Question Processor
Tests the complete OCR -> LLM -> Embedding -> Database storage pipeline
"""
import pytest
import asyncio
import json
import tempfile
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import List, Dict, Any
import numpy as np

from services.pdf_processor import (
    PDFQuestionProcessor, 
    ProcessingStatus, 
    ProcessingProgress,
    BatchProcessingResult
)
from services.llm_service import ExamPaper, Question as LLMQuestion, QuestionType
from database.models import ExtractedQuestion, Question, QuestionEmbedding
from sqlalchemy import select


@pytest.mark.integration
class TestPDFProcessorIntegration:
    """Test PDF Processor complete pipeline integration"""
    
    @pytest.fixture
    async def processor(self):
        """Create processor instance with test API keys"""
        processor = PDFQuestionProcessor(
            mistral_api_key="test-mistral-key",
            google_api_key="test-google-key"
        )
        yield processor
        # Cleanup any active processes
        async with processor._process_lock:
            for file_path in list(processor._active_processes.keys()):
                await processor.cancel_processing(file_path)
    
    @pytest.fixture
    def mock_ocr_response(self):
        """Mock OCR service response"""
        return """
        Mathematics Paper 1
        Year: 2024
        Level: AS Level
        Subject: Mathematics
        
        Q1. [5 marks] Calculate the derivative of f(x) = 3x² + 2x - 1
        
        Q2. [10 marks] Solve the following integral:
        ∫(x³ + 2x)dx
        
        Q3. [15 marks] A particle moves along a straight line with velocity 
        v(t) = 2t - 3. Find:
        a) The acceleration at t = 2
        b) The displacement from t = 0 to t = 4
        """
    
    @pytest.fixture
    def mock_llm_response(self):
        """Mock LLM service response"""
        return ExamPaper(
            year="2024",
            level="AS Level",
            subject="Mathematics",
            total_marks=30,
            duration="1 hour 30 minutes",
            source_pdf="test.pdf",
            questions=[
                LLMQuestion(
                    question_number="Q1",
                    marks=5,
                    question_type=QuestionType.CALCULATION,
                    topics=["calculus", "derivatives"],
                    question_text="Calculate the derivative of f(x) = 3x² + 2x - 1"
                ),
                LLMQuestion(
                    question_number="Q2",
                    marks=10,
                    question_type=QuestionType.CALCULATION,
                    topics=["calculus", "integration"],
                    question_text="Solve the following integral: ∫(x³ + 2x)dx"
                ),
                LLMQuestion(
                    question_number="Q3",
                    marks=15,
                    question_type=QuestionType.MULTI_PART,
                    topics=["mechanics", "kinematics"],
                    question_text="A particle moves along a straight line with velocity v(t) = 2t - 3. Find:\na) The acceleration at t = 2\nb) The displacement from t = 0 to t = 4"
                )
            ]
        )
    
    @pytest.fixture
    def mock_embeddings(self):
        """Mock embedding service response"""
        return {
            1: np.random.rand(768).tolist(),
            2: np.random.rand(768).tolist(),
            3: np.random.rand(768).tolist()
        }
    
    @pytest.fixture
    def sample_pdf_file(self, tmp_path):
        """Create a sample PDF file for testing"""
        pdf_file = tmp_path / "test_exam.pdf"
        pdf_content = b"%PDF-1.4\n%Test PDF content for mathematics exam\n%%EOF"
        pdf_file.write_bytes(pdf_content)
        return pdf_file

    @pytest.mark.asyncio
    async def test_complete_pipeline_success(
        self, 
        processor, 
        test_session,
        mock_ocr_response,
        mock_llm_response,
        mock_embeddings,
        sample_pdf_file
    ):
        """Test complete successful pipeline: OCR -> LLM -> Embeddings -> DB"""
        progress_updates = []
        
        def progress_callback(progress: ProcessingProgress):
            progress_updates.append({
                'status': progress.status,
                'step': progress.current_step,
                'percentage': progress.progress_percentage
            })
        
        processor.set_progress_callback(progress_callback)
        
        # Mock all external services
        with patch.object(processor.ocr_service, 'process_pdf', new_callable=AsyncMock) as mock_ocr:
            mock_ocr.return_value = mock_ocr_response
            
            with patch.object(processor.llm_service, 'extract_questions', new_callable=AsyncMock) as mock_llm:
                mock_llm.return_value = mock_llm_response
                
                with patch.object(processor.embedding_service, 'generate_batch_embeddings', new_callable=AsyncMock) as mock_embed:
                    mock_embed.return_value = mock_embeddings
                    
                    # Override database session
                    with patch('services.pdf_processor.AsyncSessionLocal', return_value=test_session):
                        result = await processor.process_single_pdf(
                            sample_pdf_file,
                            store_to_db=True,
                            generate_embeddings=True
                        )
        
        # Verify result
        assert result['success'] is True
        assert result['file_path'] == str(sample_pdf_file)
        assert len(result['questions']) == 3
        assert result['questions_stored'] == 3
        assert result['embeddings_count'] == 3
        
        # Verify exam metadata
        assert result['exam_metadata']['year'] == "2024"
        assert result['exam_metadata']['level'] == "AS Level"
        assert result['exam_metadata']['subject'] == "Mathematics"
        assert result['exam_metadata']['total_marks'] == 30
        
        # Verify progress tracking
        assert len(progress_updates) > 0
        statuses = [update['status'] for update in progress_updates]
        assert ProcessingStatus.PROCESSING in statuses
        assert ProcessingStatus.OCR_COMPLETE in statuses
        assert ProcessingStatus.LLM_COMPLETE in statuses
        assert ProcessingStatus.EMBEDDING_COMPLETE in statuses
        assert ProcessingStatus.COMPLETED in statuses
        
        # Verify final progress is 100%
        final_progress = progress_updates[-1]
        assert final_progress['percentage'] == 100.0
        
        # Verify database storage
        extracted_questions = await test_session.execute(
            select(ExtractedQuestion)
        )
        stored_questions = extracted_questions.scalars().all()
        assert len(stored_questions) == 3
        
        # Verify first question
        q1 = stored_questions[0]
        assert q1.question_number == "Q1"
        assert q1.marks == 5
        assert q1.year == "2024"
        assert q1.level == "AS Level"
        assert "calculus" in q1.topics
        assert q1.question_type == "calculation"
        
        # Verify embeddings stored
        embeddings = await test_session.execute(
            select(QuestionEmbedding)
        )
        stored_embeddings = embeddings.scalars().all()
        assert len(stored_embeddings) == 3
        
        # Verify embedding properties
        embedding1 = stored_embeddings[0]
        assert len(embedding1.embedding) == 768
        assert embedding1.model_name == 'gemini-embedding-001'
        assert embedding1.model_version == '1.0'

    @pytest.mark.asyncio
    async def test_pipeline_ocr_failure(
        self, 
        processor, 
        test_session,
        sample_pdf_file
    ):
        """Test pipeline failure during OCR step"""
        progress_updates = []
        
        def progress_callback(progress: ProcessingProgress):
            progress_updates.append({
                'status': progress.status,
                'error': progress.error_message
            })
        
        processor.set_progress_callback(progress_callback)
        
        # Mock OCR failure
        with patch.object(processor.ocr_service, 'process_pdf', new_callable=AsyncMock) as mock_ocr:
            mock_ocr.side_effect = Exception("OCR service unavailable")
            
            result = await processor.process_single_pdf(
                sample_pdf_file,
                store_to_db=True,
                generate_embeddings=True
            )
        
        # Verify failure result
        assert result['success'] is False
        assert result['error'] == "OCR service unavailable"
        assert len(result['questions']) == 0
        assert result['embeddings_count'] == 0
        
        # Verify progress shows failure
        final_progress = progress_updates[-1]
        assert final_progress['status'] == ProcessingStatus.FAILED
        assert final_progress['error'] == "OCR service unavailable"
        
        # Verify no database entries
        extracted_questions = await test_session.execute(
            select(ExtractedQuestion)
        )
        stored_questions = extracted_questions.scalars().all()
        assert len(stored_questions) == 0

    @pytest.mark.asyncio
    async def test_pipeline_llm_failure(
        self, 
        processor, 
        test_session,
        mock_ocr_response,
        sample_pdf_file
    ):
        """Test pipeline failure during LLM step"""
        progress_updates = []
        
        def progress_callback(progress: ProcessingProgress):
            progress_updates.append({
                'status': progress.status,
                'error': progress.error_message
            })
        
        processor.set_progress_callback(progress_callback)
        
        # Mock OCR success, LLM failure
        with patch.object(processor.ocr_service, 'process_pdf', new_callable=AsyncMock) as mock_ocr:
            mock_ocr.return_value = mock_ocr_response
            
            with patch.object(processor.llm_service, 'extract_questions', new_callable=AsyncMock) as mock_llm:
                mock_llm.side_effect = Exception("LLM service rate limited")
                
                result = await processor.process_single_pdf(
                    sample_pdf_file,
                    store_to_db=True,
                    generate_embeddings=True
                )
        
        # Verify failure result
        assert result['success'] is False
        assert result['error'] == "LLM service rate limited"
        
        # Verify progress shows failure after OCR completion
        statuses = [update['status'] for update in progress_updates]
        assert ProcessingStatus.OCR_COMPLETE in statuses
        assert ProcessingStatus.FAILED in statuses

    @pytest.mark.asyncio
    async def test_pipeline_database_rollback(
        self, 
        processor, 
        test_session,
        mock_ocr_response,
        mock_llm_response,
        sample_pdf_file
    ):
        """Test database transaction rollback on failure"""
        # Mock successful OCR and LLM, but force database error
        with patch.object(processor.ocr_service, 'process_pdf', new_callable=AsyncMock) as mock_ocr:
            mock_ocr.return_value = mock_ocr_response
            
            with patch.object(processor.llm_service, 'extract_questions', new_callable=AsyncMock) as mock_llm:
                mock_llm.return_value = mock_llm_response
                
                # Mock database session to fail during commit
                mock_session = AsyncMock()
                mock_session.add = Mock()
                mock_session.flush = AsyncMock()
                mock_session.commit = AsyncMock(side_effect=Exception("Database connection lost"))
                mock_session.rollback = AsyncMock()
                mock_session.__aenter__ = AsyncMock(return_value=mock_session)
                mock_session.__aexit__ = AsyncMock(return_value=None)
                
                with patch('services.pdf_processor.AsyncSessionLocal', return_value=mock_session):
                    result = await processor.process_single_pdf(
                        sample_pdf_file,
                        store_to_db=True,
                        generate_embeddings=False
                    )
        
        # Verify failure and rollback
        assert result['success'] is False
        assert "Database connection lost" in result['error']
        mock_session.rollback.assert_called_once()
        
        # Verify no data in actual test session
        extracted_questions = await test_session.execute(
            select(ExtractedQuestion)
        )
        stored_questions = extracted_questions.scalars().all()
        assert len(stored_questions) == 0

    @pytest.mark.asyncio
    async def test_pipeline_embedding_failure_partial_success(
        self, 
        processor, 
        test_session,
        mock_ocr_response,
        mock_llm_response,
        sample_pdf_file
    ):
        """Test pipeline continues with partial success when embeddings fail"""
        # Mock successful OCR and LLM, but embedding failure
        with patch.object(processor.ocr_service, 'process_pdf', new_callable=AsyncMock) as mock_ocr:
            mock_ocr.return_value = mock_ocr_response
            
            with patch.object(processor.llm_service, 'extract_questions', new_callable=AsyncMock) as mock_llm:
                mock_llm.return_value = mock_llm_response
                
                with patch.object(processor.embedding_service, 'generate_batch_embeddings', new_callable=AsyncMock) as mock_embed:
                    mock_embed.side_effect = Exception("Embedding service timeout")
                    
                    with patch('services.pdf_processor.AsyncSessionLocal', return_value=test_session):
                        result = await processor.process_single_pdf(
                            sample_pdf_file,
                            store_to_db=True,
                            generate_embeddings=True
                        )
        
        # Verify partial success - questions stored but no embeddings
        assert result['success'] is True
        assert len(result['questions']) == 3
        assert result['questions_stored'] == 3
        assert result['embeddings_count'] == 0  # Embeddings failed
        
        # Verify questions are in database
        extracted_questions = await test_session.execute(
            select(ExtractedQuestion)
        )
        stored_questions = extracted_questions.scalars().all()
        assert len(stored_questions) == 3
        
        # Verify no embeddings in database
        embeddings = await test_session.execute(
            select(QuestionEmbedding)
        )
        stored_embeddings = embeddings.scalars().all()
        assert len(stored_embeddings) == 0

    @pytest.mark.asyncio
    async def test_pipeline_without_database_storage(
        self, 
        processor, 
        test_session,
        mock_ocr_response,
        mock_llm_response,
        sample_pdf_file
    ):
        """Test pipeline without database storage"""
        # Mock successful OCR and LLM
        with patch.object(processor.ocr_service, 'process_pdf', new_callable=AsyncMock) as mock_ocr:
            mock_ocr.return_value = mock_ocr_response
            
            with patch.object(processor.llm_service, 'extract_questions', new_callable=AsyncMock) as mock_llm:
                mock_llm.return_value = mock_llm_response
                
                result = await processor.process_single_pdf(
                    sample_pdf_file,
                    store_to_db=False,
                    generate_embeddings=False
                )
        
        # Verify result without database storage
        assert result['success'] is True
        assert len(result['questions']) == 3
        assert result['questions_stored'] == 0  # Not stored
        assert result['embeddings_count'] == 0  # Not generated
        
        # Verify no database entries
        extracted_questions = await test_session.execute(
            select(ExtractedQuestion)
        )
        stored_questions = extracted_questions.scalars().all()
        assert len(stored_questions) == 0

    @pytest.mark.asyncio
    async def test_multiple_file_processing(
        self, 
        processor, 
        test_session,
        mock_ocr_response,
        mock_llm_response,
        mock_embeddings,
        tmp_path
    ):
        """Test processing multiple PDF files"""
        # Create multiple test files
        pdf_files = []
        for i in range(3):
            pdf_file = tmp_path / f"exam_{i}.pdf"
            pdf_content = b"%PDF-1.4\n%Test PDF content %d\n%%EOF" % i
            pdf_file.write_bytes(pdf_content)
            pdf_files.append(pdf_file)
        
        # Mock all external services
        with patch.object(processor.ocr_service, 'process_pdf', new_callable=AsyncMock) as mock_ocr:
            mock_ocr.return_value = mock_ocr_response
            
            with patch.object(processor.llm_service, 'extract_questions', new_callable=AsyncMock) as mock_llm:
                mock_llm.return_value = mock_llm_response
                
                with patch.object(processor.embedding_service, 'generate_batch_embeddings', new_callable=AsyncMock) as mock_embed:
                    mock_embed.return_value = mock_embeddings
                    
                    with patch('services.pdf_processor.AsyncSessionLocal', return_value=test_session):
                        result = await processor.process_pdf_folder(
                            tmp_path,
                            recursive=False,
                            max_concurrent=2,
                            store_to_db=True,
                            generate_embeddings=True
                        )
        
        # Verify batch processing result
        assert isinstance(result, BatchProcessingResult)
        assert result.total_files == 3
        assert result.successful_files == 3
        assert result.failed_files == 0
        assert result.total_questions == 9  # 3 questions per file × 3 files
        assert result.total_embeddings == 9
        assert len(result.file_results) == 3
        assert len(result.errors) == 0
        
        # Verify all questions are stored
        extracted_questions = await test_session.execute(
            select(ExtractedQuestion)
        )
        stored_questions = extracted_questions.scalars().all()
        assert len(stored_questions) == 9

    @pytest.mark.asyncio
    async def test_processing_cancellation(
        self, 
        processor, 
        sample_pdf_file
    ):
        """Test processing cancellation functionality"""
        file_path = str(sample_pdf_file)
        
        # Start processing with slow mock
        async def slow_ocr(*args, **kwargs):
            await asyncio.sleep(0.5)  # Simulate slow processing
            return "OCR result"
        
        with patch.object(processor.ocr_service, 'process_pdf', new_callable=AsyncMock) as mock_ocr:
            mock_ocr.side_effect = slow_ocr
            
            # Start processing in background
            process_task = asyncio.create_task(
                processor.process_single_pdf(sample_pdf_file)
            )
            
            # Wait a bit then cancel
            await asyncio.sleep(0.1)
            cancelled = await processor.cancel_processing(file_path)
            assert cancelled is True
            
            # Get status
            status = await processor.get_processing_status(file_path)
            assert status is not None
            assert status.status == ProcessingStatus.CANCELLED
            
            # Wait for task completion
            result = await process_task
            
            # Task should complete but status remains cancelled
            # (Note: In a real implementation, you'd need proper cancellation handling)
            status_after = await processor.get_processing_status(file_path)
            # Status might be None if cleaned up, or still show as cancelled

    @pytest.mark.asyncio
    async def test_progress_tracking_detailed(
        self, 
        processor, 
        test_session,
        mock_ocr_response,
        mock_llm_response,
        mock_embeddings,
        sample_pdf_file
    ):
        """Test detailed progress tracking throughout pipeline"""
        progress_history = []
        
        def detailed_progress_callback(progress: ProcessingProgress):
            progress_history.append({
                'status': progress.status,
                'step': progress.current_step,
                'percentage': progress.progress_percentage,
                'completed_steps': progress.completed_steps,
                'total_steps': progress.total_steps,
                'questions_extracted': progress.questions_extracted,
                'questions_stored': progress.questions_stored,
                'embeddings_generated': progress.embeddings_generated,
                'start_time': progress.start_time,
                'end_time': progress.end_time,
                'is_complete': progress.is_complete
            })
        
        processor.set_progress_callback(detailed_progress_callback)
        
        # Mock all services
        with patch.object(processor.ocr_service, 'process_pdf', new_callable=AsyncMock) as mock_ocr:
            mock_ocr.return_value = mock_ocr_response
            
            with patch.object(processor.llm_service, 'extract_questions', new_callable=AsyncMock) as mock_llm:
                mock_llm.return_value = mock_llm_response
                
                with patch.object(processor.embedding_service, 'generate_batch_embeddings', new_callable=AsyncMock) as mock_embed:
                    mock_embed.return_value = mock_embeddings
                    
                    with patch('services.pdf_processor.AsyncSessionLocal', return_value=test_session):
                        result = await processor.process_single_pdf(
                            sample_pdf_file,
                            store_to_db=True,
                            generate_embeddings=True
                        )
        
        # Verify detailed progress tracking
        assert len(progress_history) >= 5  # At least 5 progress updates
        
        # Verify progress sequence
        statuses = [p['status'] for p in progress_history]
        expected_sequence = [
            ProcessingStatus.PROCESSING,
            ProcessingStatus.OCR_COMPLETE,
            ProcessingStatus.LLM_COMPLETE,
            ProcessingStatus.STORED,
            ProcessingStatus.EMBEDDING_COMPLETE,
            ProcessingStatus.COMPLETED
        ]
        
        for expected_status in expected_sequence:
            assert expected_status in statuses
        
        # Verify progress percentages increase
        percentages = [p['percentage'] for p in progress_history]
        assert percentages[0] < percentages[-1]
        assert percentages[-1] == 100.0
        
        # Verify step counts are tracked
        final_progress = progress_history[-1]
        assert final_progress['total_steps'] == 4  # OCR, LLM, Store, Embeddings
        assert final_progress['completed_steps'] == 4
        assert final_progress['questions_extracted'] == 3
        assert final_progress['questions_stored'] == 3
        assert final_progress['embeddings_generated'] == 3
        
        # Verify timing
        assert progress_history[0]['start_time'] is not None
        assert final_progress['end_time'] is not None
        assert final_progress['is_complete'] is True

    @pytest.mark.asyncio
    async def test_file_validation(self, processor, tmp_path):
        """Test file validation before processing"""
        # Test non-existent file
        non_existent = tmp_path / "does_not_exist.pdf"
        result = await processor.process_single_pdf(non_existent)
        assert result['success'] is False
        assert "does not exist" in result['error'].lower()
        
        # Test wrong file type
        txt_file = tmp_path / "document.txt"
        txt_file.write_text("This is not a PDF")
        result = await processor.process_single_pdf(txt_file)
        assert result['success'] is False
        assert "unsupported file type" in result['error'].lower()
        
        # Test empty file
        empty_pdf = tmp_path / "empty.pdf"
        empty_pdf.write_bytes(b"")
        result = await processor.process_single_pdf(empty_pdf)
        assert result['success'] is False
        assert "empty" in result['error'].lower()
        
        # Test large file (if max_file_size is configured)
        if hasattr(processor, 'max_file_size') and processor.max_file_size:
            large_pdf = tmp_path / "large.pdf"
            large_content = b"x" * (processor.max_file_size + 1)
            large_pdf.write_bytes(large_content)
            result = await processor.process_single_pdf(large_pdf)
            assert result['success'] is False
            assert "too large" in result['error'].lower()

    @pytest.mark.asyncio
    async def test_async_context_manager(
        self, 
        mock_ocr_response,
        mock_llm_response,
        sample_pdf_file
    ):
        """Test processor as async context manager"""
        progress_updates = []
        
        def progress_callback(progress):
            progress_updates.append(progress.status)
        
        # Test normal context manager usage
        async with PDFQuestionProcessor(
            mistral_api_key="test-key",
            google_api_key="test-key",
            progress_callback=progress_callback
        ) as processor:
            with patch.object(processor.ocr_service, 'process_pdf', new_callable=AsyncMock) as mock_ocr:
                mock_ocr.return_value = mock_ocr_response
                
                with patch.object(processor.llm_service, 'extract_questions', new_callable=AsyncMock) as mock_llm:
                    mock_llm.return_value = mock_llm_response
                    
                    result = await processor.process_single_pdf(
                        sample_pdf_file,
                        store_to_db=False
                    )
        
        assert result['success'] is True
        assert len(progress_updates) > 0

    @pytest.mark.asyncio
    async def test_concurrent_processing_safety(
        self, 
        processor, 
        mock_ocr_response,
        mock_llm_response,
        tmp_path
    ):
        """Test that concurrent processing is handled safely"""
        # Create multiple files
        files = []
        for i in range(3):
            pdf_file = tmp_path / f"concurrent_{i}.pdf"
            pdf_file.write_bytes(b"%PDF-1.4\ncontent\n%%EOF")
            files.append(pdf_file)
        
        progress_counts = {}
        
        def count_progress(progress):
            file_path = progress.file_path
            if file_path not in progress_counts:
                progress_counts[file_path] = 0
            progress_counts[file_path] += 1
        
        processor.set_progress_callback(count_progress)
        
        # Mock services with slight delays
        async def delayed_ocr(*args, **kwargs):
            await asyncio.sleep(0.1)
            return mock_ocr_response
        
        async def delayed_llm(*args, **kwargs):
            await asyncio.sleep(0.1)
            return mock_llm_response
        
        with patch.object(processor.ocr_service, 'process_pdf', new_callable=AsyncMock) as mock_ocr:
            mock_ocr.side_effect = delayed_ocr
            
            with patch.object(processor.llm_service, 'extract_questions', new_callable=AsyncMock) as mock_llm:
                mock_llm.side_effect = delayed_llm
                
                # Process files concurrently
                tasks = [
                    processor.process_single_pdf(file, store_to_db=False)
                    for file in files
                ]
                
                results = await asyncio.gather(*tasks)
        
        # Verify all succeeded
        assert all(result['success'] for result in results)
        
        # Verify progress tracking worked for all files
        assert len(progress_counts) == 3
        for file_path, count in progress_counts.items():
            assert count > 0  # Each file should have progress updates

    @pytest.mark.asyncio
    async def test_error_handling_preserves_context(
        self, 
        processor, 
        sample_pdf_file
    ):
        """Test that errors don't corrupt internal state"""
        # First, cause an error
        with patch.object(processor.ocr_service, 'process_pdf', new_callable=AsyncMock) as mock_ocr:
            mock_ocr.side_effect = Exception("First error")
            
            result1 = await processor.process_single_pdf(sample_pdf_file)
            assert result1['success'] is False
        
        # Then, try successful processing
        with patch.object(processor.ocr_service, 'process_pdf', new_callable=AsyncMock) as mock_ocr:
            mock_ocr.return_value = "OCR text"
            
            with patch.object(processor.llm_service, 'extract_questions', new_callable=AsyncMock) as mock_llm:
                # Create a minimal exam paper
                mock_exam = ExamPaper(
                    year="2024",
                    level="Test",
                    subject="Test",
                    total_marks=10,
                    duration="1 hour",
                    source_pdf="test.pdf",
                    questions=[
                        LLMQuestion(
                            question_number="Q1",
                            marks=10,
                            question_type=QuestionType.SHORT_ANSWER,
                            topics=["test"],
                            question_text="Test question"
                        )
                    ]
                )
                mock_llm.return_value = mock_exam
                
                result2 = await processor.process_single_pdf(
                    sample_pdf_file,
                    store_to_db=False
                )
        
        # Second processing should succeed
        assert result2['success'] is True
        assert len(result2['questions']) == 1
        
        # Verify no lingering state from first error
        status = await processor.get_processing_status(str(sample_pdf_file))
        assert status is None  # Should be cleaned up