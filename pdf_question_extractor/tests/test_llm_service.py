"""
Unit tests for LLM Service (Gemini)
"""
import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import httpx
import json
from google import genai
from google.genai import types
from services.llm_service import GeminiLLMService, Question, ExamPaper


@pytest.mark.unit
class TestGeminiLLMService:
    """Test Gemini LLM Service functionality"""
    
    @pytest.fixture
    def llm_service(self):
        """Create LLM service instance"""
        return GeminiLLMService(api_key="test-api-key")
    
    @pytest.fixture
    def sample_ocr_text(self):
        """Sample OCR text for testing"""
        return """
        Mathematics Paper 1
        Year: 2024
        Level: AS Level
        
        Q1. [5 marks] Calculate the derivative of f(x) = 3x² + 2x - 1
        
        Q2. [10 marks] Solve the following integral:
        ∫(x³ + 2x)dx
        
        Q3. [15 marks] A particle moves along a straight line with velocity 
        v(t) = 2t - 3. Find:
        a) The acceleration at t = 2
        b) The displacement from t = 0 to t = 4
        """
    
    @pytest.fixture
    def mock_gemini_response(self):
        """Mock successful Gemini response"""
        response_data = {
            "exam_info": {
                "year": "2024",
                "level": "AS Level",
                "subject": "Mathematics",
                "paper": "Paper 1"
            },
            "questions": [
                {
                    "question_number": "Q1",
                    "marks": 5,
                    "question_type": "calculation",
                    "topics": ["calculus", "derivatives"],
                    "question_text": "Calculate the derivative of f(x) = 3x² + 2x - 1"
                },
                {
                    "question_number": "Q2",
                    "marks": 10,
                    "question_type": "calculation",
                    "topics": ["calculus", "integration"],
                    "question_text": "Solve the following integral: ∫(x³ + 2x)dx"
                },
                {
                    "question_number": "Q3",
                    "marks": 15,
                    "question_type": "multi_part",
                    "topics": ["mechanics", "kinematics"],
                    "question_text": "A particle moves along a straight line with velocity v(t) = 2t - 3. Find:\na) The acceleration at t = 2\nb) The displacement from t = 0 to t = 4"
                }
            ]
        }
        
        # Create a mock response that simulates Gemini's response
        mock_response = Mock()
        mock_response.text = json.dumps(response_data)
        return mock_response
    
    @pytest.mark.asyncio
    async def test_extract_questions_success(self, llm_service, sample_ocr_text, mock_gemini_response):
        """Test successful question extraction"""
        # Mock the generate_content method
        with patch.object(llm_service.client.models, 'generate_content') as mock_generate:
            # Make it return an async function
            async def async_return(*args, **kwargs):
                return mock_gemini_response
            
            mock_generate.return_value = async_return()
            
            # Mock rate limiter
            with patch.object(llm_service.rate_limiter, 'acquire', new_callable=AsyncMock):
                result = await llm_service.extract_questions(sample_ocr_text, "test.pdf")
        
        # Verify result
        assert isinstance(result, ExamPaper)
        assert result.exam_info.year == "2024"
        assert result.exam_info.level == "AS Level"
        assert result.exam_info.subject == "Mathematics"
        assert len(result.questions) == 3
        
        # Verify first question
        q1 = result.questions[0]
        assert q1.question_number == "Q1"
        assert q1.marks == 5
        assert q1.question_type == "calculation"
        assert "calculus" in q1.topics
    
    @pytest.mark.asyncio
    async def test_extract_questions_chunking(self, llm_service):
        """Test text chunking for large documents"""
        # Create large text (>50k characters)
        large_text = "Q1. Test question\n" * 10000  # Will be >50k chars
        
        mock_response = Mock()
        mock_response.text = json.dumps({
            "exam_info": {"year": "2024", "level": "AS", "subject": "Test", "paper": "1"},
            "questions": [{"question_number": "Q1", "marks": 5, "question_type": "short", 
                          "topics": ["test"], "question_text": "Test question"}]
        })
        
        with patch.object(llm_service.client.models, 'generate_content') as mock_generate:
            async def async_return(*args, **kwargs):
                return mock_response
            
            mock_generate.return_value = async_return()
            
            with patch.object(llm_service.rate_limiter, 'acquire', new_callable=AsyncMock):
                result = await llm_service.extract_questions(large_text, "large.pdf")
        
        # Should have called generate_content multiple times for chunks
        assert mock_generate.call_count > 1
        assert isinstance(result, ExamPaper)
    
    @pytest.mark.asyncio
    async def test_extract_questions_rate_limiting(self, llm_service, sample_ocr_text, mock_gemini_response):
        """Test rate limiting functionality"""
        with patch.object(llm_service.client.models, 'generate_content') as mock_generate:
            async def async_return(*args, **kwargs):
                return mock_gemini_response
            
            mock_generate.return_value = async_return()
            
            # Mock rate limiter to track calls
            with patch.object(llm_service.rate_limiter, 'acquire', new_callable=AsyncMock) as mock_acquire:
                await llm_service.extract_questions(sample_ocr_text, "test.pdf")
        
        # Verify rate limiter was called
        mock_acquire.assert_called()
    
    @pytest.mark.asyncio
    async def test_extract_questions_retry_on_error(self, llm_service, sample_ocr_text, mock_gemini_response):
        """Test retry mechanism on API errors"""
        # Setup to fail twice then succeed
        side_effects = [
            httpx.HTTPError("Server error"),
            httpx.HTTPError("Server error"),
            mock_gemini_response
        ]
        
        with patch.object(llm_service.client.models, 'generate_content') as mock_generate:
            # Create async side effects
            async def async_side_effect(*args, **kwargs):
                effect = side_effects.pop(0)
                if isinstance(effect, Exception):
                    raise effect
                return effect
            
            mock_generate.side_effect = async_side_effect
            
            with patch.object(llm_service.rate_limiter, 'acquire', new_callable=AsyncMock):
                result = await llm_service.extract_questions(sample_ocr_text, "test.pdf")
        
        assert isinstance(result, ExamPaper)
        # Should have been called 3 times
        assert mock_generate.call_count == 3
    
    @pytest.mark.asyncio
    async def test_extract_questions_invalid_json(self, llm_service, sample_ocr_text):
        """Test handling of invalid JSON response"""
        # Create response with invalid JSON
        mock_response = Mock()
        mock_response.text = "This is not valid JSON"
        
        with patch.object(llm_service.client.models, 'generate_content') as mock_generate:
            async def async_return(*args, **kwargs):
                return mock_response
            
            mock_generate.return_value = async_return()
            
            with patch.object(llm_service.rate_limiter, 'acquire', new_callable=AsyncMock):
                # Should raise ValueError for invalid JSON
                with pytest.raises(ValueError, match="Failed to parse"):
                    await llm_service.extract_questions(sample_ocr_text, "test.pdf")
    
    @pytest.mark.asyncio
    async def test_extract_questions_missing_fields(self, llm_service, sample_ocr_text):
        """Test handling of response with missing required fields"""
        # Response missing required fields
        incomplete_response = {
            "exam_info": {
                "year": "2024"
                # Missing other required fields
            },
            "questions": [
                {
                    "question_number": "Q1"
                    # Missing other required fields
                }
            ]
        }
        
        mock_response = Mock()
        mock_response.text = json.dumps(incomplete_response)
        
        with patch.object(llm_service.client.models, 'generate_content') as mock_generate:
            async def async_return(*args, **kwargs):
                return mock_response
            
            mock_generate.return_value = async_return()
            
            with patch.object(llm_service.rate_limiter, 'acquire', new_callable=AsyncMock):
                # Should raise validation error
                with pytest.raises(ValueError):
                    await llm_service.extract_questions(sample_ocr_text, "test.pdf")
    
    @pytest.mark.asyncio
    async def test_generate_content_configuration(self, llm_service, sample_ocr_text):
        """Test that generate_content is called with correct configuration"""
        mock_response = Mock()
        mock_response.text = json.dumps({
            "exam_info": {"year": "2024", "level": "AS", "subject": "Test", "paper": "1"},
            "questions": []
        })
        
        call_args_capture = {}
        
        async def capture_args(*args, **kwargs):
            call_args_capture.update(kwargs)
            return mock_response
        
        with patch.object(llm_service.client.models, 'generate_content', side_effect=capture_args):
            with patch.object(llm_service.rate_limiter, 'acquire', new_callable=AsyncMock):
                await llm_service.extract_questions(sample_ocr_text, "test.pdf")
        
        # Verify configuration
        assert call_args_capture['model'] == 'models/gemini-2.0-flash-exp'
        assert 'contents' in call_args_capture
        assert call_args_capture['config']['response_mime_type'] == 'application/json'
        assert 'response_schema' in call_args_capture['config']
    
    @pytest.mark.asyncio
    async def test_question_validation(self, llm_service):
        """Test Question model validation"""
        # Valid question
        valid_question = Question(
            question_number="Q1",
            marks=10,
            question_type="essay",
            topics=["history", "world_war_2"],
            question_text="Discuss the causes of World War 2."
        )
        assert valid_question.marks == 10
        assert len(valid_question.topics) == 2
        
        # Invalid question (negative marks)
        with pytest.raises(ValueError):
            Question(
                question_number="Q1",
                marks=-5,  # Invalid
                question_type="essay",
                topics=["test"],
                question_text="Test question"
            )
        
        # Empty topics list
        with pytest.raises(ValueError):
            Question(
                question_number="Q1",
                marks=10,
                question_type="essay",
                topics=[],  # Invalid - empty list
                question_text="Test question"
            )
    
    @pytest.mark.asyncio
    async def test_exam_paper_to_dict(self, llm_service):
        """Test ExamPaper to_dict method"""
        exam_paper = ExamPaper(
            exam_info={
                "year": "2024",
                "level": "AS Level",
                "subject": "Mathematics",
                "paper": "Paper 1"
            },
            questions=[
                Question(
                    question_number="Q1",
                    marks=5,
                    question_type="calculation",
                    topics=["calculus"],
                    question_text="Calculate the derivative"
                )
            ]
        )
        
        result_dict = exam_paper.to_dict()
        
        assert result_dict["exam_info"]["year"] == "2024"
        assert len(result_dict["questions"]) == 1
        assert result_dict["questions"][0]["question_number"] == "Q1"
        assert result_dict["questions"][0]["marks"] == 5