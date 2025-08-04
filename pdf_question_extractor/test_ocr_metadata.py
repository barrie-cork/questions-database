#!/usr/bin/env python3
"""
Test script to verify Mistral OCR API metadata extraction
"""
import asyncio
import os
from services.ocr_service import MistralOCRService
from config import Config

async def test_ocr_metadata():
    """Test OCR with real PDF and inspect metadata"""
    # Initialize OCR service
    ocr_service = MistralOCRService(api_key=Config.MISTRAL_API_KEY)
    
    # Test PDF path
    pdf_path = "/app/tests/test_pdfs/June 2022 QP.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"Error: Test PDF not found at {pdf_path}")
        return
    
    print(f"Processing PDF: {pdf_path}")
    print(f"File size: {os.path.getsize(pdf_path) / 1024 / 1024:.2f} MB")
    
    try:
        # Process PDF with OCR
        markdown_text = await ocr_service.process_pdf(pdf_path, include_images=False)
        
        print(f"\nOCR Results:")
        print(f"- Text length: {len(markdown_text)} characters")
        print(f"- First 500 chars:\n{markdown_text[:500]}...")
        
        # Try to get more detailed response by patching
        import json
        from unittest.mock import patch
        
        # Capture raw response
        raw_response = None
        
        def capture_response(*args, **kwargs):
            nonlocal raw_response
            # Call the real method
            response = ocr_service.client.ocr.process(*args, **kwargs)
            raw_response = response
            return response
        
        with patch.object(ocr_service.client.ocr, 'process', side_effect=capture_response):
            await ocr_service.process_pdf(pdf_path, include_images=False)
        
        if raw_response:
            print("\nRaw Response Structure:")
            print(f"- Type: {type(raw_response)}")
            print(f"- Attributes: {dir(raw_response)}")
            
            if hasattr(raw_response, '__dict__'):
                print(f"- Dict content: {raw_response.__dict__}")
            
            if hasattr(raw_response, 'model_dump'):
                print(f"- Model dump: {json.dumps(raw_response.model_dump(), indent=2)}")
        
    except Exception as e:
        print(f"Error: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_ocr_metadata())