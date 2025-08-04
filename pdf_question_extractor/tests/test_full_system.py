"""
Full system test with real PDF processing.

This test runs the complete pipeline with a real PDF file to verify:
- OCR extraction works correctly
- LLM extracts questions properly
- Embeddings are generated
- Database storage works
- API endpoints return correct data
"""

import pytest
import asyncio
import os
import json
from pathlib import Path
from typing import Dict, Any

import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from database.session import AsyncSessionLocal
from database.models import ExtractedQuestion, Question, QuestionEmbedding
from services.pdf_processor import PDFQuestionProcessor
from config import Config


@pytest.mark.system
@pytest.mark.asyncio
class TestFullSystem:
    """Full system integration test with real PDF"""
    
    @pytest.fixture
    def test_pdf_path(self):
        """Path to test PDF file"""
        return "/mnt/d/Python/Projects/Dave/questions_pdf_to_sheet/pdf_question_extractor/tests/test_pdfs/June 2022 QP.pdf"
    
    @pytest.fixture
    def api_base_url(self):
        """Base URL for API calls"""
        return "http://localhost:8000"
    
    async def test_complete_pdf_processing_pipeline(self, test_pdf_path, api_base_url):
        """Test complete pipeline from PDF upload to export"""
        
        # Skip if API keys not configured
        if not os.getenv("MISTRAL_API_KEY") or not os.getenv("GOOGLE_API_KEY"):
            pytest.skip("API keys not configured for system test")
        
        # 1. Upload PDF via API
        print("\n1. Uploading PDF file...")
        async with httpx.AsyncClient() as client:
            with open(test_pdf_path, "rb") as f:
                files = {"file": ("June 2022 QP.pdf", f, "application/pdf")}
                response = await client.post(
                    f"{api_base_url}/api/upload",
                    files=files,
                    timeout=300  # 5 minutes for processing
                )
        
        assert response.status_code == 200
        upload_data = response.json()
        print(f"   ✓ Upload successful: {upload_data['filename']}")
        print(f"   ✓ Questions extracted: {upload_data['questions_extracted']}")
        print(f"   ✓ Questions stored: {upload_data['questions_stored']}")
        
        # 2. Verify questions in database
        print("\n2. Verifying database storage...")
        async with AsyncSessionLocal() as session:
            # Check extracted questions
            result = await session.execute(
                select(ExtractedQuestion).where(
                    ExtractedQuestion.source_pdf == upload_data['filename']
                )
            )
            extracted_questions = result.scalars().all()
            
            assert len(extracted_questions) > 0
            print(f"   ✓ Found {len(extracted_questions)} questions in database")
            
            # Check embeddings
            result = await session.execute(
                select(QuestionEmbedding).join(ExtractedQuestion).where(
                    ExtractedQuestion.source_pdf == upload_data['filename']
                )
            )
            embeddings = result.scalars().all()
            
            assert len(embeddings) > 0
            print(f"   ✓ Found {len(embeddings)} embeddings in database")
            
            # Verify embedding dimensions
            if embeddings:
                assert len(embeddings[0].embedding) == 768
                print(f"   ✓ Embeddings have correct dimension: 768")
        
        # 3. Test API endpoints
        print("\n3. Testing API endpoints...")
        
        # Get questions list
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{api_base_url}/api/questions",
                params={"per_page": 10, "page": 1}
            )
        
        assert response.status_code == 200
        questions_data = response.json()
        assert questions_data["total"] > 0
        assert len(questions_data["items"]) > 0
        print(f"   ✓ Questions API returned {questions_data['total']} questions")
        
        # Get stats
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{api_base_url}/api/stats")
        
        assert response.status_code == 200
        stats_data = response.json()
        assert stats_data["total_extracted"] > 0
        print(f"   ✓ Stats API shows {stats_data['total_extracted']} extracted questions")
        
        # 4. Test question update
        print("\n4. Testing question update...")
        question_id = questions_data["items"][0]["id"]
        
        async with httpx.AsyncClient() as client:
            update_data = {
                "status": "approved",
                "edited_text": "Updated question text for testing"
            }
            response = await client.put(
                f"{api_base_url}/api/questions/{question_id}",
                json=update_data
            )
        
        assert response.status_code == 200
        updated_question = response.json()
        assert updated_question["status"] == "approved"
        assert updated_question["edited_text"] == "Updated question text for testing"
        print(f"   ✓ Question {question_id} updated successfully")
        
        # 5. Test bulk operations
        print("\n5. Testing bulk operations...")
        question_ids = [q["id"] for q in questions_data["items"][:3]]
        
        async with httpx.AsyncClient() as client:
            bulk_data = {
                "question_ids": question_ids,
                "operation": "approve"
            }
            response = await client.post(
                f"{api_base_url}/api/questions/bulk",
                json=bulk_data
            )
        
        assert response.status_code == 200
        bulk_result = response.json()
        assert bulk_result["updated"] == len(question_ids)
        print(f"   ✓ Bulk approved {bulk_result['updated']} questions")
        
        # 6. Test export
        print("\n6. Testing export functionality...")
        
        # Export as JSON
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{api_base_url}/api/export",
                params={"format": "json", "status": "approved"}
            )
        
        assert response.status_code == 200
        export_data = response.json()
        assert "questions" in export_data
        assert len(export_data["questions"]) > 0
        print(f"   ✓ JSON export returned {len(export_data['questions'])} approved questions")
        
        # Export as CSV
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{api_base_url}/api/export",
                params={"format": "csv", "status": "approved"}
            )
        
        assert response.status_code == 200
        assert "text/csv" in response.headers["content-type"]
        csv_content = response.text
        assert "question_text" in csv_content
        print(f"   ✓ CSV export successful")
        
        # 7. Test search functionality
        print("\n7. Testing search functionality...")
        
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{api_base_url}/api/questions",
                params={"search": "calculate", "per_page": 5}
            )
        
        assert response.status_code == 200
        search_results = response.json()
        print(f"   ✓ Search returned {search_results['total']} results")
        
        # 8. Verify complete pipeline
        print("\n8. Final verification...")
        print(f"   ✓ Complete pipeline test PASSED!")
        print(f"   ✓ PDF processed: {test_pdf_path}")
        print(f"   ✓ Questions extracted: {upload_data['questions_extracted']}")
        print(f"   ✓ Questions stored: {upload_data['questions_stored']}")
        print(f"   ✓ Embeddings generated: {upload_data['embeddings_generated']}")
        
        return {
            "pdf_processed": test_pdf_path,
            "questions_extracted": upload_data['questions_extracted'],
            "questions_stored": upload_data['questions_stored'],
            "embeddings_generated": upload_data['embeddings_generated'],
            "test_status": "PASSED"
        }
    
    async def test_direct_processor_with_real_pdf(self, test_pdf_path):
        """Test PDF processor directly without API"""
        
        # Skip if API keys not configured
        if not os.getenv("MISTRAL_API_KEY") or not os.getenv("GOOGLE_API_KEY"):
            pytest.skip("API keys not configured for system test")
        
        print("\n\nDirect Processor Test")
        print("=" * 50)
        
        # Create processor
        processor = PDFQuestionProcessor()
        
        # Progress tracking
        progress_updates = []
        
        async def progress_callback(progress):
            progress_updates.append(progress)
            print(f"   {progress.current_step} ({progress.completed_steps}/{progress.total_steps})")
        
        # Process PDF
        print(f"\nProcessing: {test_pdf_path}")
        result = await processor.process_pdf(test_pdf_path, progress_callback)
        
        # Verify results
        assert result["questions_extracted"] > 0
        assert result["questions_stored"] > 0
        assert result["embeddings_generated"] > 0
        
        print(f"\n✓ Processing complete!")
        print(f"  - Questions extracted: {result['questions_extracted']}")
        print(f"  - Questions stored: {result['questions_stored']}")
        print(f"  - Embeddings generated: {result['embeddings_generated']}")
        
        # Verify progress tracking
        assert len(progress_updates) > 0
        assert progress_updates[-1].status.value == "completed"
        
        return result


if __name__ == "__main__":
    # Run the test directly
    import asyncio
    
    async def run_test():
        test = TestFullSystem()
        pdf_path = test.test_pdf_path()
        api_url = test.api_base_url()
        
        try:
            # Run API test
            result = await test.test_complete_pdf_processing_pipeline(pdf_path, api_url)
            print("\n" + "=" * 50)
            print("FULL SYSTEM TEST COMPLETED SUCCESSFULLY")
            print("=" * 50)
            print(json.dumps(result, indent=2))
        except Exception as e:
            print(f"\nERROR: {e}")
            
            # Try direct processor test as fallback
            print("\nTrying direct processor test...")
            result = await test.test_direct_processor_with_real_pdf(pdf_path)
    
    asyncio.run(run_test())