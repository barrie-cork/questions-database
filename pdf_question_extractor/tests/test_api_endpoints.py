"""
Comprehensive API endpoint tests for the FastAPI application.

Tests all API endpoints including:
- GET /api/health
- GET /api/stats  
- POST /api/upload
- GET /api/questions (with pagination)
- PUT /api/questions/{id}
- POST /api/questions/bulk (approve/reject/delete)
- POST /api/questions/save
- GET /api/export (CSV and JSON formats)

Uses FastAPI TestClient with mocked services and database operations.
"""

import pytest
import json
import asyncio
import tempfile
import io
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import Mock, AsyncMock, patch, MagicMock

import httpx
from fastapi.testclient import TestClient
from fastapi import status
from sqlalchemy.ext.asyncio import AsyncSession

from app import app
from database.models import ExtractedQuestion, Question, QuestionEmbedding
from api.schemas.requests import (
    QuestionStatusEnum, BulkOperationEnum, ExportFormatEnum, 
    QuestionTypeEnum
)
from api.schemas.responses import (
    HealthResponse, StatsResponse, UploadResponse, QuestionsListResponse,
    QuestionResponse, BulkOperationResponse, SaveApprovedResponse,
    ExportResponse
)


class TestHealthEndpoint:
    """Test health check endpoint"""
    
    def test_health_check_success(self, test_client: TestClient):
        """Test successful health check"""
        response = test_client.get("/health")
        
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert data["status"] in ["healthy", "degraded"]
        assert "timestamp" in data
        assert data["version"] == "1.0.0"
        assert "database_connected" in data
        assert "services" in data
        
        # Check services structure
        services = data["services"]
        assert "database" in services
        assert "ocr_service" in services
        assert "llm_service" in services
        assert "embedding_service" in services
    
    @patch('app.engine.begin')
    def test_health_check_database_failure(self, mock_engine, test_client: TestClient):
        """Test health check with database failure"""
        # Mock database connection failure
        mock_conn = AsyncMock()
        mock_conn.execute.side_effect = Exception("Database connection failed")
        mock_engine.return_value.__aenter__.return_value = mock_conn
        
        response = test_client.get("/health")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] == "degraded"
        assert data["database_connected"] is False
        assert data["services"]["database"] == "unhealthy"


class TestStatsEndpoint:
    """Test statistics endpoint"""
    
    @pytest.fixture  
    async def setup_test_data(self, test_session: AsyncSession):
        """Setup test data for stats"""
        # Create test extracted questions
        extracted_q1 = ExtractedQuestion(
            question_text="Test question 1",
            source_pdf="test1.pdf",
            status="pending",
            year="2024",
            level="AS Level",
            question_type="multiple_choice"
        )
        extracted_q2 = ExtractedQuestion(
            question_text="Test question 2", 
            source_pdf="test2.pdf",
            status="approved",
            year="2024",
            level="A Level",
            question_type="essay"
        )
        
        # Create test permanent questions
        permanent_q1 = Question(
            question_text="Permanent question 1",
            source_pdf="permanent1.pdf",
            year="2023",
            level="AS Level", 
            question_type="calculation"
        )
        
        test_session.add_all([extracted_q1, extracted_q2, permanent_q1])
        await test_session.commit()
        
        return {
            "extracted": [extracted_q1, extracted_q2],
            "permanent": [permanent_q1]
        }
    
    async def test_stats_endpoint(self, test_client: TestClient, setup_test_data):
        """Test statistics endpoint"""
        response = test_client.get("/api/stats")
        
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert "total_extracted_questions" in data
        assert "total_approved_questions" in data
        assert "total_permanent_questions" in data
        assert "questions_by_status" in data
        assert "questions_by_year" in data
        assert "questions_by_level" in data
        assert "questions_by_type" in data
        assert "recent_processing_activity" in data
        
        # Verify data structure
        assert isinstance(data["questions_by_status"], dict)
        assert isinstance(data["questions_by_year"], dict)
        assert isinstance(data["questions_by_level"], dict)
        assert isinstance(data["questions_by_type"], dict)
        assert isinstance(data["recent_processing_activity"], list)
    
    def test_stats_endpoint_error(self, test_client: TestClient):
        """Test stats endpoint with database error"""
        with patch('api.routes.get_db') as mock_get_db:
            mock_session = AsyncMock()
            mock_session.execute.side_effect = Exception("Database error")
            mock_get_db.return_value = mock_session
            
            response = test_client.get("/api/stats")
            
            assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
            assert "Failed to get statistics" in response.json()["message"]


class TestUploadEndpoint:
    """Test file upload endpoint"""
    
    def create_test_pdf_file(self) -> bytes:
        """Create a test PDF file content"""
        return b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n>>\nendobj\nxref\n0 1\n0000000000 65535 f \ntrailer\n<<\n/Size 1\n/Root 1 0 R\n>>\nstartxref\n9\n%%EOF"
    
    @patch('api.routes.get_processor')
    def test_upload_single_pdf_success(self, mock_get_processor, test_client: TestClient):
        """Test successful single PDF upload"""
        # Mock processor
        mock_processor = AsyncMock()
        mock_processor.process_single_pdf = AsyncMock()
        mock_get_processor.return_value = mock_processor
        
        # Create test file
        pdf_content = self.create_test_pdf_file()
        
        response = test_client.post(
            "/api/upload",
            files=[("files", ("test.pdf", pdf_content, "application/pdf"))],
            data={
                "store_to_db": "true",
                "generate_embeddings": "true"
            }
        )
        
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert data["success"] is True
        assert "Successfully uploaded" in data["message"]
        assert "processing_id" in data
        assert len(data["files_uploaded"]) == 1
        assert data["total_files"] == 1
        
        # Verify processor was called
        mock_processor.process_single_pdf.assert_called_once()
    
    @patch('api.routes.get_processor')
    def test_upload_multiple_pdfs(self, mock_get_processor, test_client: TestClient):
        """Test multiple PDF upload"""
        # Mock processor
        mock_processor = AsyncMock()
        mock_processor.process_pdf_folder = AsyncMock()
        mock_get_processor.return_value = mock_processor
        
        # Create test files
        pdf_content = self.create_test_pdf_file()
        
        response = test_client.post(
            "/api/upload",  
            files=[
                ("files", ("test1.pdf", pdf_content, "application/pdf")),
                ("files", ("test2.pdf", pdf_content, "application/pdf"))
            ],
            data={
                "store_to_db": "true",
                "generate_embeddings": "false",
                "max_concurrent": "2"
            }
        )
        
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert data["success"] is True
        assert data["total_files"] == 2
        
        # Verify processor was called with correct parameters
        mock_processor.process_pdf_folder.assert_called_once()
    
    def test_upload_no_pdf_files(self, test_client: TestClient):
        """Test upload with no PDF files"""
        response = test_client.post(
            "/api/upload",
            files=[("files", ("test.txt", b"not a pdf", "text/plain"))]
        )
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "No valid PDF files provided" in response.json()["detail"]
    
    def test_upload_no_files(self, test_client: TestClient):
        """Test upload with no files"""
        response = test_client.post("/api/upload")
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    @patch('api.routes.get_processor')
    def test_upload_request_validation(self, mock_get_processor, test_client: TestClient):
        """Test upload request validation"""
        mock_processor = AsyncMock()
        mock_get_processor.return_value = mock_processor
        
        pdf_content = self.create_test_pdf_file()
        
        # Test with invalid max_concurrent
        response = test_client.post(
            "/api/upload",
            files=[("files", ("test.pdf", pdf_content, "application/pdf"))],
            data={"max_concurrent": "15"}  # Max is 10
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    @patch('api.routes.get_processor')  
    def test_upload_processor_error(self, mock_get_processor, test_client: TestClient):
        """Test upload with processor error"""
        mock_get_processor.side_effect = Exception("Processor initialization failed")
        
        pdf_content = self.create_test_pdf_file()
        
        response = test_client.post(
            "/api/upload",
            files=[("files", ("test.pdf", pdf_content, "application/pdf"))]
        )
        
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert "Upload failed" in response.json()["detail"]


class TestQuestionsEndpoint:
    """Test questions listing endpoint"""
    
    @pytest.fixture
    async def setup_questions(self, test_session: AsyncSession):
        """Setup test questions"""
        questions = []
        
        # Create extracted questions
        for i in range(25):  # More than one page
            q = ExtractedQuestion(
                question_number=f"Q{i+1}",
                marks=5 + i,
                year="2024" if i % 2 == 0 else "2023",
                level="AS Level" if i % 3 == 0 else "A Level",
                topics=["math", "calculus"] if i % 2 == 0 else ["physics"],
                question_type="multiple_choice" if i % 2 == 0 else "essay",
                question_text=f"Test question {i+1}",
                source_pdf=f"test{i+1}.pdf",
                status="approved" if i % 3 == 0 else "pending"
            )
            questions.append(q)
            test_session.add(q)
        
        await test_session.commit()
        return questions
    
    async def test_get_questions_default(self, test_client: TestClient, setup_questions):
        """Test getting questions with default parameters"""
        response = test_client.get("/api/questions")
        
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert "questions" in data
        assert "total" in data
        assert "page" in data
        assert "per_page" in data
        assert "total_pages" in data
        assert "has_next" in data
        assert "has_prev" in data
        
        # Check pagination defaults
        assert data["page"] == 1
        assert data["per_page"] == 20
        assert data["has_prev"] is False
        assert len(data["questions"]) <= 20
    
    async def test_get_questions_pagination(self, test_client: TestClient, setup_questions):
        """Test questions pagination"""
        # First page
        response = test_client.get("/api/questions?page=1&per_page=10")
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert data["page"] == 1
        assert data["per_page"] == 10
        assert len(data["questions"]) == 10
        assert data["has_next"] is True
        assert data["has_prev"] is False
        
        # Second page
        response = test_client.get("/api/questions?page=2&per_page=10")
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert data["page"] == 2
        assert data["has_prev"] is True
    
    async def test_get_questions_filters(self, test_client: TestClient, setup_questions):
        """Test questions with filters"""
        # Filter by status
        response = test_client.get("/api/questions?status_filter=approved")
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        for question in data["questions"]:
            assert question["status"] == "approved"
        
        # Filter by year
        response = test_client.get("/api/questions?year_filter=2024")
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        for question in data["questions"]:
            assert question["year"] == "2024"
        
        # Filter by level
        response = test_client.get("/api/questions?level_filter=AS Level")
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        for question in data["questions"]:
            assert question["level"] == "AS Level"
    
    async def test_get_questions_search(self, test_client: TestClient, setup_questions):
        """Test questions search functionality"""
        response = test_client.get("/api/questions?search=question 1")
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        # Should find questions containing "question 1"
        found_q1 = any("question 1" in q["question_text"] for q in data["questions"])
        assert found_q1
    
    def test_get_questions_validation_errors(self, test_client: TestClient):
        """Test questions endpoint validation"""
        # Invalid page
        response = test_client.get("/api/questions?page=0")
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        
        # Invalid per_page
        response = test_client.get("/api/questions?per_page=200")
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    async def test_get_questions_table_parameter(self, test_client: TestClient, setup_questions):
        """Test questions endpoint with table parameter"""
        # Test extracted table (default)
        response = test_client.get("/api/questions?table=extracted")
        assert response.status_code == status.HTTP_200_OK
        
        # Test permanent table
        response = test_client.get("/api/questions?table=permanent")
        assert response.status_code == status.HTTP_200_OK
    
    def test_get_questions_database_error(self, test_client: TestClient):
        """Test questions endpoint with database error"""
        with patch('api.routes.get_db') as mock_get_db:
            mock_session = AsyncMock()
            mock_session.execute.side_effect = Exception("Database error")
            mock_get_db.return_value = mock_session
            
            response = test_client.get("/api/questions")
            
            assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
            assert "Failed to retrieve questions" in response.json()["detail"]


class TestUpdateQuestionEndpoint:
    """Test single question update endpoint"""
    
    @pytest.fixture
    async def setup_question(self, test_session: AsyncSession):
        """Setup a test question"""
        question = ExtractedQuestion(
            id=1,
            question_number="Q1",
            marks=10,
            year="2024",
            level="AS Level",
            topics=["math"],
            question_type="multiple_choice",
            question_text="Original question text",
            source_pdf="test.pdf",
            status="pending"
        )
        test_session.add(question)
        await test_session.commit()
        await test_session.refresh(question)
        return question
    
    async def test_update_question_success(self, test_client: TestClient, setup_question):
        """Test successful question update"""
        question_id = setup_question.id
        
        update_data = {
            "question_text": "Updated question text",
            "marks": 15,
            "status": "approved",
            "topics": ["math", "algebra"]
        }
        
        response = test_client.put(f"/api/questions/{question_id}", json=update_data)
        
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert data["id"] == question_id
        assert data["question_text"] == "Updated question text"
        assert data["marks"] == 15
        assert data["status"] == "approved"
        assert "algebra" in data["topics"]
    
    async def test_update_question_partial(self, test_client: TestClient, setup_question):
        """Test partial question update"""
        question_id = setup_question.id
        
        update_data = {
            "marks": 20
        }
        
        response = test_client.put(f"/api/questions/{question_id}", json=update_data)
        
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert data["marks"] == 20
        assert data["question_text"] == "Original question text"  # Unchanged
    
    def test_update_question_not_found(self, test_client: TestClient):
        """Test updating non-existent question"""
        response = test_client.put(
            "/api/questions/99999", 
            json={"marks": 10}
        )
        
        assert response.status_code == status.HTTP_404_NOT_FOUND
        assert "not found" in response.json()["detail"]
    
    def test_update_question_validation_error(self, test_client: TestClient):
        """Test question update with validation errors"""
        # Invalid marks (negative)
        response = test_client.put(
            "/api/questions/1",
            json={"marks": -5}
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    async def test_update_question_table_parameter(self, test_client: TestClient, setup_question):
        """Test updating with table parameter"""
        question_id = setup_question.id
        
        # Update in extracted table
        response = test_client.put(
            f"/api/questions/{question_id}?table=extracted",
            json={"marks": 25}
        )
        
        assert response.status_code == status.HTTP_200_OK
        
        # Test permanent table (would fail since question doesn't exist there)
        response = test_client.put(
            f"/api/questions/{question_id}?table=permanent",
            json={"marks": 30}
        )
        
        assert response.status_code == status.HTTP_404_NOT_FOUND
    
    def test_update_question_database_error(self, test_client: TestClient):
        """Test question update with database error"""
        with patch('api.routes.get_db') as mock_get_db:
            mock_session = AsyncMock()
            mock_session.execute.side_effect = Exception("Database error") 
            mock_get_db.return_value = mock_session
            
            response = test_client.put("/api/questions/1", json={"marks": 10})
            
            assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
            assert "Failed to update question" in response.json()["detail"]


class TestBulkOperationsEndpoint:
    """Test bulk operations endpoint"""
    
    @pytest.fixture
    async def setup_bulk_questions(self, test_session: AsyncSession):
        """Setup questions for bulk operations"""
        questions = []
        for i in range(5):
            q = ExtractedQuestion(
                question_number=f"Q{i+1}",
                marks=10,
                question_text=f"Bulk test question {i+1}",
                source_pdf=f"bulk{i+1}.pdf",
                status="pending"
            )
            questions.append(q)
            test_session.add(q)
        
        await test_session.commit()
        for q in questions:
            await test_session.refresh(q)
        return questions
    
    async def test_bulk_approve(self, test_client: TestClient, setup_bulk_questions):
        """Test bulk approve operation"""
        question_ids = [q.id for q in setup_bulk_questions[:3]]
        
        bulk_data = {
            "question_ids": question_ids,
            "operation": "approve"
        }
        
        response = test_client.post("/api/questions/bulk", json=bulk_data)
        
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert data["success"] is True
        assert data["affected_count"] == 3
        assert len(data["failed_ids"]) == 0
        assert "3 questions affected" in data["message"]
    
    async def test_bulk_reject(self, test_client: TestClient, setup_bulk_questions):
        """Test bulk reject operation"""
        question_ids = [q.id for q in setup_bulk_questions[:2]]
        
        bulk_data = {
            "question_ids": question_ids,
            "operation": "reject"
        }
        
        response = test_client.post("/api/questions/bulk", json=bulk_data)
        
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert data["success"] is True
        assert data["affected_count"] == 2
    
    async def test_bulk_delete(self, test_client: TestClient, setup_bulk_questions):
        """Test bulk delete operation"""
        question_ids = [setup_bulk_questions[0].id]
        
        bulk_data = {
            "question_ids": question_ids,
            "operation": "delete"
        }
        
        response = test_client.post("/api/questions/bulk", json=bulk_data)
        
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert data["success"] is True
        assert data["affected_count"] == 1
    
    async def test_bulk_update_status(self, test_client: TestClient, setup_bulk_questions):
        """Test bulk status update operation"""
        question_ids = [q.id for q in setup_bulk_questions[:2]]
        
        bulk_data = {
            "question_ids": question_ids,
            "operation": "update_status",
            "new_status": "approved"
        }
        
        response = test_client.post("/api/questions/bulk", json=bulk_data)
        
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert data["success"] is True
        assert data["affected_count"] == 2
    
    def test_bulk_operation_validation_errors(self, test_client: TestClient):
        """Test bulk operation validation"""
        # Missing question_ids
        response = test_client.post("/api/questions/bulk", json={
            "operation": "approve"
        })
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        
        # Empty question_ids
        response = test_client.post("/api/questions/bulk", json={
            "question_ids": [],
            "operation": "approve"
        })
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        
        # Update status without new_status
        response = test_client.post("/api/questions/bulk", json={
            "question_ids": [1, 2],
            "operation": "update_status"
        })
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_bulk_operation_not_found(self, test_client: TestClient):
        """Test bulk operation with non-existent questions"""
        bulk_data = {
            "question_ids": [99999, 99998],
            "operation": "approve"
        }
        
        response = test_client.post("/api/questions/bulk", json=bulk_data)
        
        assert response.status_code == status.HTTP_404_NOT_FOUND
        assert "No questions found" in response.json()["detail"]
    
    async def test_bulk_operation_table_parameter(self, test_client: TestClient, setup_bulk_questions):
        """Test bulk operations with table parameter"""
        question_ids = [setup_bulk_questions[0].id]
        
        bulk_data = {
            "question_ids": question_ids,
            "operation": "approve"
        }
        
        # Test extracted table
        response = test_client.post("/api/questions/bulk?table=extracted", json=bulk_data)
        assert response.status_code == status.HTTP_200_OK
        
        # Test permanent table (would fail since questions don't exist there)
        response = test_client.post("/api/questions/bulk?table=permanent", json=bulk_data)
        assert response.status_code == status.HTTP_404_NOT_FOUND
    
    def test_bulk_operation_database_error(self, test_client: TestClient):
        """Test bulk operation with database error"""
        with patch('api.routes.get_db') as mock_get_db:
            mock_session = AsyncMock()
            mock_session.execute.side_effect = Exception("Database error")
            mock_get_db.return_value = mock_session
            
            bulk_data = {
                "question_ids": [1, 2],
                "operation": "approve"
            }
            
            response = test_client.post("/api/questions/bulk", json=bulk_data)
            
            assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
            assert "Bulk operation failed" in response.json()["detail"]


class TestSaveApprovedEndpoint:
    """Test save approved questions endpoint"""
    
    @pytest.fixture
    async def setup_approved_questions(self, test_session: AsyncSession):
        """Setup approved questions for saving"""
        questions = []
        
        # Create approved questions
        for i in range(3):
            q = ExtractedQuestion(
                question_number=f"Q{i+1}",
                marks=10 + i,
                year="2024",
                level="AS Level",
                topics=["test"],
                question_type="multiple_choice",
                question_text=f"Approved question {i+1}",
                source_pdf=f"approved{i+1}.pdf",
                status="approved"
            )
            questions.append(q)
            test_session.add(q)
        
        # Create a pending question (should not be saved)
        pending_q = ExtractedQuestion(
            question_text="Pending question",
            source_pdf="pending.pdf",
            status="pending"
        )
        questions.append(pending_q)
        test_session.add(pending_q)
        
        await test_session.commit()
        for q in questions:
            await test_session.refresh(q)
        return questions
    
    async def test_save_all_approved_questions(self, test_client: TestClient, setup_approved_questions):
        """Test saving all approved questions"""
        save_data = {
            "clear_extracted": True
        }
        
        response = test_client.post("/api/questions/save", json=save_data)
        
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert data["success"] is True
        assert data["saved_count"] == 3  # Only approved questions
        assert data["cleared_count"] == 3
        assert data["failed_count"] == 0
        assert "Successfully saved 3 approved questions" in data["message"]
    
    async def test_save_specific_approved_questions(self, test_client: TestClient, setup_approved_questions):
        """Test saving specific approved questions"""
        # Get IDs of first 2 approved questions
        approved_questions = [q for q in setup_approved_questions if q.status == "approved"]
        question_ids = [approved_questions[0].id, approved_questions[1].id]
        
        save_data = {
            "question_ids": question_ids,
            "clear_extracted": False
        }
        
        response = test_client.post("/api/questions/save", json=save_data)
        
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert data["success"] is True
        assert data["saved_count"] == 2
        assert data["cleared_count"] == 0  # clear_extracted was False
    
    async def test_save_no_clear_extracted(self, test_client: TestClient, setup_approved_questions):
        """Test saving without clearing extracted questions"""
        save_data = {
            "clear_extracted": False
        }
        
        response = test_client.post("/api/questions/save", json=save_data)
        
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert data["cleared_count"] == 0
    
    def test_save_no_approved_questions(self, test_client: TestClient):
        """Test saving when no approved questions exist"""
        save_data = {
            "clear_extracted": True
        }
        
        response = test_client.post("/api/questions/save", json=save_data)
        
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert data["success"] is True
        assert data["saved_count"] == 0
        assert "No approved questions found" in data["message"]
    
    def test_save_approved_database_error(self, test_client: TestClient):
        """Test save approved with database error"""
        with patch('api.routes.get_db') as mock_get_db:
            mock_session = AsyncMock()
            mock_session.execute.side_effect = Exception("Database error")
            mock_get_db.return_value = mock_session
            
            save_data = {"clear_extracted": True}
            
            response = test_client.post("/api/questions/save", json=save_data)
            
            assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
            assert "Failed to save approved questions" in response.json()["detail"]


class TestExportEndpoint:
    """Test export questions endpoint"""
    
    @pytest.fixture
    async def setup_export_questions(self, test_session: AsyncSession):
        """Setup questions for export testing"""
        questions = []
        
        # Create questions with different attributes for filtering
        for i in range(10):
            q = Question(
                question_number=f"Q{i+1}",
                marks=5 + i,
                year="2024" if i % 2 == 0 else "2023",
                level="AS Level" if i % 3 == 0 else "A Level",
                topics=["math"] if i % 2 == 0 else ["physics"],
                question_type="multiple_choice" if i % 2 == 0 else "essay",
                question_text=f"Export test question {i+1}",
                source_pdf=f"export{i+1}.pdf"
            )
            questions.append(q)
            test_session.add(q)
        
        await test_session.commit()
        for q in questions:
            await test_session.refresh(q)
        return questions
    
    @patch('api.routes.export_to_csv')
    @patch('pathlib.Path.stat')
    def test_export_csv_format(self, mock_stat, mock_export_csv, test_client: TestClient, setup_export_questions):
        """Test CSV export"""
        # Mock file operations
        mock_stat.return_value.st_size = 1024
        mock_csv_output = io.StringIO("id,question_text\n1,Test question")
        mock_export_csv.return_value = mock_csv_output
        
        response = test_client.get("/api/export?format=csv")
        
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert data["success"] is True
        assert data["format"] == "csv"
        assert data["filename"].endswith(".csv")
        assert "download_url" in data
        assert data["file_size"] == 1024
        assert data["record_count"] > 0
    
    @patch('pathlib.Path.stat')
    async def test_export_json_format(self, mock_stat, test_client: TestClient, setup_export_questions):
        """Test JSON export"""
        mock_stat.return_value.st_size = 2048
        
        response = test_client.get("/api/export?format=json")
        
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert data["success"] is True
        assert data["format"] == "json"
        assert data["filename"].endswith(".json")
        assert data["file_size"] == 2048
    
    def test_export_unsupported_format(self, test_client: TestClient):
        """Test export with unsupported format"""
        response = test_client.get("/api/export?format=excel")
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "not yet implemented" in response.json()["detail"]
    
    @patch('pathlib.Path.stat')
    async def test_export_with_filters(self, mock_stat, test_client: TestClient, setup_export_questions):
        """Test export with filters"""
        mock_stat.return_value.st_size = 512
        
        # Test with year filter
        response = test_client.get("/api/export?format=json&year_filter=2024")
        assert response.status_code == status.HTTP_200_OK
        
        # Test with level filter
        response = test_client.get("/api/export?format=json&level_filter=AS Level")
        assert response.status_code == status.HTTP_200_OK
        
        # Test with question type filter  
        response = test_client.get("/api/export?format=json&question_type_filter=multiple_choice")
        assert response.status_code == status.HTTP_200_OK
    
    @patch('pathlib.Path.stat')
    async def test_export_specific_questions(self, mock_stat, test_client: TestClient, setup_export_questions):
        """Test export with specific question IDs"""
        mock_stat.return_value.st_size = 256
        
        question_ids = [setup_export_questions[0].id, setup_export_questions[1].id]
        
        response = test_client.get(f"/api/export?format=json&question_ids={','.join(map(str, question_ids))}")
        
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert data["record_count"] <= 2
    
    @patch('pathlib.Path.stat')
    async def test_export_metadata_options(self, mock_stat, test_client: TestClient, setup_export_questions):
        """Test export with metadata options"""
        mock_stat.return_value.st_size = 1024
        
        # Test with metadata included
        response = test_client.get("/api/export?format=json&include_metadata=true")
        assert response.status_code == status.HTTP_200_OK
        
        # Test without metadata
        response = test_client.get("/api/export?format=json&include_metadata=false")
        assert response.status_code == status.HTTP_200_OK
    
    def test_export_database_error(self, test_client: TestClient):
        """Test export with database error"""
        with patch('api.routes.get_db') as mock_get_db:
            mock_session = AsyncMock()
            mock_session.execute.side_effect = Exception("Database error")
            mock_get_db.return_value = mock_session
            
            response = test_client.get("/api/export?format=json")
            
            assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
            assert "Export failed" in response.json()["detail"]


class TestDownloadEndpoint:
    """Test file download endpoint"""
    
    @patch('pathlib.Path.exists')
    @patch('fastapi.responses.FileResponse')
    def test_download_existing_file(self, mock_file_response, mock_exists, test_client: TestClient):
        """Test downloading existing file"""
        mock_exists.return_value = True
        mock_file_response.return_value = Mock()
        
        response = test_client.get("/api/download/test_export.csv")
        
        # The actual response depends on FileResponse, but we can check the call
        mock_file_response.assert_called_once()
    
    @patch('pathlib.Path.exists')
    def test_download_nonexistent_file(self, mock_exists, test_client: TestClient):
        """Test downloading non-existent file"""
        mock_exists.return_value = False
        
        response = test_client.get("/api/download/nonexistent.csv")
        
        assert response.status_code == status.HTTP_404_NOT_FOUND
        assert "File not found" in response.json()["detail"]


class TestCORSHeaders:
    """Test CORS headers are properly set"""
    
    def test_cors_headers_options(self, test_client: TestClient):
        """Test CORS preflight request"""
        response = test_client.options(
            "/api/questions",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
                "Access-Control-Request-Headers": "Content-Type"
            }
        )
        
        assert response.status_code == status.HTTP_200_OK
        assert "access-control-allow-origin" in response.headers
        assert "access-control-allow-methods" in response.headers
        assert "access-control-allow-headers" in response.headers
    
    def test_cors_headers_get_request(self, test_client: TestClient):
        """Test CORS headers on GET request"""
        response = test_client.get(
            "/api/questions",
            headers={"Origin": "http://localhost:3000"}
        )
        
        assert "access-control-allow-origin" in response.headers


class TestErrorHandling:
    """Test error handling across endpoints"""
    
    def test_validation_error_format(self, test_client: TestClient):
        """Test validation error response format"""
        # Send invalid data to trigger validation error
        response = test_client.post("/api/questions/bulk", json={
            "operation": "invalid_operation"
        })
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        
        data = response.json()
        assert data["error"] is True
        assert "message" in data
        assert "errors" in data
        assert "timestamp" in data
    
    def test_http_error_format(self, test_client: TestClient):
        """Test HTTP error response format"""
        response = test_client.get("/api/questions/99999")
        
        # This should return a 404 or 405 (method not allowed)
        assert response.status_code in [status.HTTP_404_NOT_FOUND, status.HTTP_405_METHOD_NOT_ALLOWED]
        
        data = response.json()
        assert "message" in data


class TestRequestValidation:
    """Test request validation across endpoints"""
    
    def test_upload_request_validation(self, test_client: TestClient):
        """Test upload request validation"""
        # Test invalid max_concurrent value
        pdf_content = b"%PDF-1.4\n%test\n%%EOF"
        
        response = test_client.post(
            "/api/upload",
            files=[("files", ("test.pdf", pdf_content, "application/pdf"))],
            data={"max_concurrent": "15"}  # Max allowed is 10
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_question_update_validation(self, test_client: TestClient):
        """Test question update validation"""
        # Test invalid marks (negative)
        response = test_client.put(
            "/api/questions/1",
            json={"marks": -10}
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_pagination_validation(self, test_client: TestClient):
        """Test pagination parameter validation"""
        # Test invalid page number
        response = test_client.get("/api/questions?page=0")
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        
        # Test invalid per_page
        response = test_client.get("/api/questions?per_page=200")
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


class TestAsyncPatterns:
    """Test proper async patterns are used"""
    
    @patch('api.routes.get_db')
    async def test_async_database_operations(self, mock_get_db, test_client: TestClient):
        """Test that database operations are properly async"""
        mock_session = AsyncMock()
        mock_result = AsyncMock()
        mock_result.scalar.return_value = 0
        mock_session.execute.return_value = mock_result
        mock_get_db.return_value = mock_session
        
        response = test_client.get("/api/stats")
        
        # Verify async database session was used
        mock_session.execute.assert_called()
    
    @patch('api.routes.get_processor')
    async def test_async_processing_operations(self, mock_get_processor, test_client: TestClient):
        """Test that processing operations are properly async"""
        mock_processor = AsyncMock()
        mock_processor.process_single_pdf = AsyncMock()
        mock_get_processor.return_value = mock_processor
        
        pdf_content = b"%PDF-1.4\n%test\n%%EOF"
        
        response = test_client.post(
            "/api/upload",
            files=[("files", ("test.pdf", pdf_content, "application/pdf"))]
        )
        
        # Verify async processor method was set up for background task
        assert response.status_code == status.HTTP_200_OK


# Integration test combining multiple endpoints
class TestIntegrationScenarios:
    """Integration tests combining multiple endpoints"""
    
    @pytest.fixture
    async def full_scenario_setup(self, test_session: AsyncSession):
        """Setup for full integration scenario"""
        # Create extracted questions
        questions = []
        for i in range(5):
            q = ExtractedQuestion(
                question_number=f"Q{i+1}",
                marks=10,
                year="2024",
                level="AS Level", 
                topics=["integration_test"],
                question_type="multiple_choice",
                question_text=f"Integration test question {i+1}",
                source_pdf=f"integration{i+1}.pdf",
                status="pending"
            )
            questions.append(q)
            test_session.add(q)
        
        await test_session.commit()
        for q in questions:
            await test_session.refresh(q)
        return questions
    
    async def test_full_workflow_scenario(self, test_client: TestClient, full_scenario_setup):
        """Test complete workflow: extract -> review -> approve -> save -> export"""
        questions = full_scenario_setup
        
        # 1. Get initial questions list
        response = test_client.get("/api/questions")
        assert response.status_code == status.HTTP_200_OK
        initial_data = response.json()
        assert initial_data["total"] == 5
        
        # 2. Update a few questions
        question_ids = [q.id for q in questions[:3]]
        
        for qid in question_ids:
            response = test_client.put(f"/api/questions/{qid}", json={
                "marks": 15,
                "status": "approved"
            })
            assert response.status_code == status.HTTP_200_OK
        
        # 3. Bulk approve remaining questions
        remaining_ids = [q.id for q in questions[3:]]
        bulk_data = {
            "question_ids": remaining_ids,
            "operation": "approve"
        }
        
        response = test_client.post("/api/questions/bulk", json=bulk_data)
        assert response.status_code == status.HTTP_200_OK
        
        # 4. Save all approved questions
        save_data = {"clear_extracted": False}
        response = test_client.post("/api/questions/save", json=save_data)
        assert response.status_code == status.HTTP_200_OK
        save_result = response.json()
        assert save_result["saved_count"] == 5
        
        # 5. Export questions
        with patch('pathlib.Path.stat') as mock_stat:
            mock_stat.return_value.st_size = 1024
            response = test_client.get("/api/export?format=json")
            assert response.status_code == status.HTTP_200_OK
            export_result = response.json()
            assert export_result["success"] is True
        
        # 6. Check final stats
        response = test_client.get("/api/stats")
        assert response.status_code == status.HTTP_200_OK
        stats_data = response.json()
        assert stats_data["total_permanent_questions"] >= 5