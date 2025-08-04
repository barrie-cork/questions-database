"""
Unit tests for OCR Service using new Mistral OCR API
"""
import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import httpx
import base64
import os
import asyncio
from services.ocr_service import MistralOCRService


@pytest.mark.unit
class TestMistralOCRService:
    """Test Mistral OCR Service functionality"""
    
    @pytest.fixture
    def ocr_service(self):
        """Create OCR service instance"""
        return MistralOCRService(api_key="test-api-key")
    
    @pytest.fixture
    def mock_ocr_response(self):
        """Mock successful Mistral OCR response"""
        mock_response = Mock()
        mock_response.content = [
            Mock(text="Sample OCR text from PDF\n\nThis is extracted content.")
        ]
        return mock_response
    
    @pytest.fixture
    def mock_ocr_client(self):
        """Mock Mistral client with OCR capabilities"""
        mock_client = Mock()
        mock_ocr = Mock()
        mock_client.ocr = mock_ocr
        mock_process = Mock()
        mock_ocr.process = mock_process
        return mock_client, mock_process
    
    @pytest.mark.asyncio
    async def test_process_pdf_file_success(self, ocr_service, mock_ocr_response, sample_pdf_path):
        """Test successful PDF processing from file"""
        # Mock the client.ocr.process method
        with patch.object(ocr_service.client.ocr, 'process', return_value=mock_ocr_response):
            # Mock run_in_executor to run synchronously
            with patch('asyncio.get_event_loop') as mock_loop:
                mock_loop.return_value.run_in_executor = AsyncMock(
                    side_effect=lambda executor, func: func()
                )
                
                result = await ocr_service.process_pdf(sample_pdf_path)
        
        # Verify
        assert result == "Sample OCR text from PDF\n\nThis is extracted content."
        ocr_service.client.ocr.process.assert_called_once()
        
        # Verify the call arguments
        call_args = ocr_service.client.ocr.process.call_args
        assert call_args[1]['model'] == 'mistral-ocr-latest'
        assert call_args[1]['document']['type'] == 'document_base64'
        assert 'document_base64' in call_args[1]['document']
        assert call_args[1]['include_image_base64'] is True
    
    @pytest.mark.asyncio
    async def test_process_pdf_url_success(self, ocr_service, mock_ocr_response):
        """Test successful PDF processing from URL"""
        pdf_url = "https://example.com/sample.pdf"
        
        # Mock the client.ocr.process method
        with patch.object(ocr_service.client.ocr, 'process', return_value=mock_ocr_response):
            # Mock run_in_executor
            with patch('asyncio.get_event_loop') as mock_loop:
                mock_loop.return_value.run_in_executor = AsyncMock(
                    side_effect=lambda executor, func: func()
                )
                
                result = await ocr_service.process_pdf_from_url(pdf_url)
        
        # Verify
        assert result == "Sample OCR text from PDF\n\nThis is extracted content."
        ocr_service.client.ocr.process.assert_called_once()
        
        # Verify URL was passed correctly
        call_args = ocr_service.client.ocr.process.call_args
        assert call_args[1]['document']['type'] == 'document_url'
        assert call_args[1]['document']['document_url'] == pdf_url
    
    @pytest.mark.asyncio
    async def test_process_pdf_file_not_found(self, ocr_service):
        """Test handling of non-existent file"""
        with pytest.raises(FileNotFoundError):
            await ocr_service.process_pdf("/non/existent/file.pdf")
    
    @pytest.mark.asyncio
    async def test_process_pdf_large_file(self, ocr_service, tmp_path):
        """Test handling of large file (>50MB)"""
        # Create a large file
        large_file = tmp_path / "large.pdf"
        large_file.write_bytes(b"x" * (51 * 1024 * 1024))  # 51MB
        
        with pytest.raises(ValueError, match="PDF file too large"):
            await ocr_service.process_pdf(str(large_file))
    
    @pytest.mark.asyncio
    async def test_retry_on_http_error(self, ocr_service, mock_ocr_response):
        """Test retry mechanism on HTTP errors"""
        # Setup mock to fail twice then succeed
        side_effects = [
            httpx.HTTPError("Server error"),
            httpx.HTTPError("Server error"),
            mock_ocr_response
        ]
        
        with patch.object(ocr_service.client.ocr, 'process', side_effect=side_effects):
            with patch('asyncio.get_event_loop') as mock_loop:
                # Make run_in_executor execute the function and propagate exceptions
                def executor_side_effect(executor, func):
                    result = func()
                    if isinstance(result, Exception):
                        raise result
                    return result
                
                mock_loop.return_value.run_in_executor = AsyncMock(
                    side_effect=executor_side_effect
                )
                
                result = await ocr_service.process_pdf_from_url("https://example.com/test.pdf")
        
        assert result == "Sample OCR text from PDF\n\nThis is extracted content."
        assert ocr_service.client.ocr.process.call_count == 3
    
    @pytest.mark.asyncio
    async def test_failed_after_max_retries(self, ocr_service):
        """Test failure after max retry attempts"""
        # Setup mock to always fail
        with patch.object(ocr_service.client.ocr, 'process', side_effect=httpx.HTTPError("Persistent error")):
            with patch('asyncio.get_event_loop') as mock_loop:
                mock_loop.return_value.run_in_executor = AsyncMock(
                    side_effect=lambda executor, func: func()
                )
                
                with pytest.raises(httpx.HTTPError):
                    await ocr_service.process_pdf_from_url("https://example.com/test.pdf")
        
        # Should have tried 3 times
        assert ocr_service.client.ocr.process.call_count == 3
    
    @pytest.mark.asyncio
    async def test_response_parsing_variations(self, ocr_service):
        """Test different response format handling"""
        test_cases = [
            # Standard response with content
            {
                "response": Mock(content=[Mock(text="Standard text")]),
                "expected": "Standard text"
            },
            # Multiple content items (take first)
            {
                "response": Mock(content=[
                    Mock(text="First content"),
                    Mock(text="Second content")
                ]),
                "expected": "First content"
            },
            # Empty content list
            {
                "response": Mock(content=[]),
                "expected": "[]"
            },
            # No content attribute
            {
                "response": Mock(spec=[]),
                "expected": str(Mock(spec=[]))
            },
            # Content without text attribute
            {
                "response": Mock(content=[Mock(spec=['other_attr'])]),
                "expected": str([Mock(spec=['other_attr'])])
            }
        ]
        
        for test_case in test_cases:
            with patch.object(ocr_service.client.ocr, 'process', return_value=test_case["response"]):
                with patch('asyncio.get_event_loop') as mock_loop:
                    mock_loop.return_value.run_in_executor = AsyncMock(
                        side_effect=lambda executor, func: func()
                    )
                    
                    result = await ocr_service.process_pdf_from_url("https://example.com/test.pdf")
            
            # For Mock objects, we need to compare the structure, not exact string
            if "Mock" in test_case["expected"]:
                assert "Mock" in result
            else:
                assert result == test_case["expected"]
    
    @pytest.mark.asyncio
    async def test_base64_encoding(self, ocr_service, mock_ocr_response, tmp_path):
        """Test proper base64 encoding of PDF files"""
        # Create a test PDF
        pdf_content = b"%PDF-1.4\n%Test PDF content\n%%EOF"
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(pdf_content)
        
        # Capture the call arguments
        call_args_capture = {}
        
        def capture_args(*args, **kwargs):
            call_args_capture.update(kwargs)
            return mock_ocr_response
        
        with patch.object(ocr_service.client.ocr, 'process', side_effect=capture_args):
            with patch('asyncio.get_event_loop') as mock_loop:
                mock_loop.return_value.run_in_executor = AsyncMock(
                    side_effect=lambda executor, func: func()
                )
                
                result = await ocr_service.process_pdf(str(pdf_file))
        
        # Verify base64 encoding was used correctly
        assert call_args_capture['document']['type'] == 'document_base64'
        expected_base64 = base64.b64encode(pdf_content).decode('utf-8')
        assert call_args_capture['document']['document_base64'] == expected_base64
    
    @pytest.mark.asyncio
    async def test_include_images_parameter(self, ocr_service, mock_ocr_response):
        """Test include_images parameter handling"""
        # Test with include_images=True (default)
        with patch.object(ocr_service.client.ocr, 'process', return_value=mock_ocr_response) as mock_process:
            with patch('asyncio.get_event_loop') as mock_loop:
                mock_loop.return_value.run_in_executor = AsyncMock(
                    side_effect=lambda executor, func: func()
                )
                
                await ocr_service.process_pdf_from_url("https://example.com/test.pdf", include_images=True)
        
        call_args = mock_process.call_args
        assert call_args[1]['include_image_base64'] is True
        
        # Test with include_images=False
        with patch.object(ocr_service.client.ocr, 'process', return_value=mock_ocr_response) as mock_process:
            with patch('asyncio.get_event_loop') as mock_loop:
                mock_loop.return_value.run_in_executor = AsyncMock(
                    side_effect=lambda executor, func: func()
                )
                
                await ocr_service.process_pdf_from_url("https://example.com/test.pdf", include_images=False)
        
        call_args = mock_process.call_args
        assert call_args[1]['include_image_base64'] is False
    
    @pytest.mark.asyncio
    async def test_file_size_check(self, ocr_service, tmp_path):
        """Test file size validation"""
        # Create files of different sizes
        small_file = tmp_path / "small.pdf"
        small_file.write_bytes(b"x" * 1024)  # 1KB
        
        medium_file = tmp_path / "medium.pdf" 
        medium_file.write_bytes(b"x" * (10 * 1024 * 1024))  # 10MB
        
        # These should work fine
        mock_response = Mock(content=[Mock(text="Content")])
        with patch.object(ocr_service.client.ocr, 'process', return_value=mock_response):
            with patch('asyncio.get_event_loop') as mock_loop:
                mock_loop.return_value.run_in_executor = AsyncMock(
                    side_effect=lambda executor, func: func()
                )
                
                # Small file should work
                result = await ocr_service.process_pdf(str(small_file))
                assert result == "Content"
                
                # Medium file should work
                result = await ocr_service.process_pdf(str(medium_file))
                assert result == "Content"