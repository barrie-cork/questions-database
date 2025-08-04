import os
import base64
import asyncio
import aiofiles
from mistralai import Mistral
from typing import Optional, Dict, Any
import logging
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import httpx

logger = logging.getLogger(__name__)

class MistralOCRService:
    """Service for OCR processing using Mistral OCR API"""
    
    def __init__(self, api_key: str):
        """Initialize Mistral OCR client"""
        self.client = Mistral(api_key=api_key)
        self.model = "mistral-ocr-latest"
        
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException))
    )
    async def process_pdf(self, pdf_path: str, include_images: bool = True) -> str:
        """
        Process PDF file using Mistral OCR API
        
        Args:
            pdf_path: Path to PDF file
            include_images: Whether to include base64 encoded images in output
            
        Returns:
            Structured Markdown text with extracted content
        """
        try:
            # Check file size (max 50MB)
            file_size = os.path.getsize(pdf_path)
            if file_size > 50 * 1024 * 1024:  # 50MB
                raise ValueError(f"PDF file too large: {file_size / 1024 / 1024:.2f}MB (max 50MB)")
            
            # Read PDF file
            async with aiofiles.open(pdf_path, 'rb') as f:
                pdf_content = await f.read()
            
            # Encode to base64
            pdf_base64 = base64.b64encode(pdf_content).decode('utf-8')
            
            logger.info(f"Processing PDF: {pdf_path} ({file_size / 1024 / 1024:.2f}MB)")
            
            # Call Mistral OCR API (wrapped in executor to prevent blocking)
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.ocr.process(
                    model=self.model,
                    document={
                        "type": "document_url",
                        "document_url": f"data:application/pdf;base64,{pdf_base64}"
                    },
                    include_image_base64=include_images
                )
            )
            
            # Extract text content with robust parsing
            markdown_text = ""
            if hasattr(response, 'content') and response.content:
                try:
                    # Try to extract text from first content item
                    if len(response.content) > 0 and hasattr(response.content[0], 'text'):
                        markdown_text = response.content[0].text
                    else:
                        # Fallback to string representation
                        logger.warning("Response has content but unexpected structure")
                        markdown_text = str(response.content)
                except (AttributeError, IndexError) as e:
                    logger.warning(f"Error parsing response content: {e}")
                    markdown_text = str(response)
            else:
                # Last resort: convert entire response to string
                logger.warning("Response has no content attribute, using string representation")
                markdown_text = str(response)
            
            logger.info(f"OCR completed for {pdf_path}, extracted {len(markdown_text)} characters")
            
            return markdown_text
            
        except Exception as e:
            logger.error(f"OCR processing failed for {pdf_path}: {str(e)}")
            raise
    
    async def process_pdf_from_url(self, pdf_url: str, include_images: bool = True) -> str:
        """
        Process PDF from URL using Mistral OCR API
        
        Args:
            pdf_url: URL to PDF file
            include_images: Whether to include base64 encoded images in output
            
        Returns:
            Structured Markdown text with extracted content
        """
        try:
            logger.info(f"Processing PDF from URL: {pdf_url}")
            
            # Call Mistral OCR API with URL (wrapped in executor to prevent blocking)
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.ocr.process(
                    model=self.model,
                    document={
                        "type": "document_url",
                        "document_url": pdf_url
                    },
                    include_image_base64=include_images
                )
            )
            
            # Extract text content with robust parsing
            markdown_text = ""
            if hasattr(response, 'content') and response.content:
                try:
                    # Try to extract text from first content item
                    if len(response.content) > 0 and hasattr(response.content[0], 'text'):
                        markdown_text = response.content[0].text
                    else:
                        # Fallback to string representation
                        logger.warning("Response has content but unexpected structure")
                        markdown_text = str(response.content)
                except (AttributeError, IndexError) as e:
                    logger.warning(f"Error parsing response content: {e}")
                    markdown_text = str(response)
            else:
                # Last resort: convert entire response to string
                logger.warning("Response has no content attribute, using string representation")
                markdown_text = str(response)
            
            logger.info(f"OCR completed for URL, extracted {len(markdown_text)} characters")
            
            return markdown_text
            
        except Exception as e:
            logger.error(f"OCR processing failed for URL {pdf_url}: {str(e)}")
            raise
    
    def extract_metadata(self, ocr_response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract metadata from OCR response
        
        Args:
            ocr_response: Raw OCR API response
            
        Returns:
            Dictionary containing extracted metadata
        """
        metadata = {
            "page_count": 0,
            "has_images": False,
            "has_tables": False,
            "has_equations": False,
            "language": "unknown"
        }
        
        # Parse response to extract metadata
        # This would depend on the actual Mistral OCR response structure
        
        return metadata