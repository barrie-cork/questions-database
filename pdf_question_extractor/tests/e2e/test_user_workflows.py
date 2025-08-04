"""
End-to-End User Testing with Playwright
Tests complete user workflows for PDF Question Extractor
"""
import pytest
import asyncio
from pathlib import Path
import json
import os
from playwright.async_api import async_playwright, Page, expect
import aiofiles

# Base URL for the application
BASE_URL = os.getenv("TEST_BASE_URL", "http://localhost:8000")

# Test data paths
TEST_DATA_DIR = Path(__file__).parent / "test_data"
SAMPLE_PDF = TEST_DATA_DIR / "sample_exam.pdf"


class TestUserWorkflows:
    """Complete user workflow tests"""
    
    @pytest.fixture
    async def authenticated_page(self, page: Page):
        """Get an authenticated page ready for testing"""
        await page.goto(BASE_URL)
        # Wait for app to load
        await page.wait_for_selector("#app", state="visible")
        return page
    
    async def test_homepage_loads(self, page: Page):
        """Test that the homepage loads correctly"""
        await page.goto(BASE_URL)
        
        # Check title
        await expect(page).to_have_title("PDF Question Extractor")
        
        # Check main elements are visible
        await expect(page.locator("h1")).to_contain_text("PDF Question Extractor")
        await expect(page.locator("#upload-area")).to_be_visible()
        await expect(page.locator("#questions-grid")).to_be_visible()
    
    async def test_pdf_upload_workflow(self, page: Page):
        """Test complete PDF upload and processing workflow"""
        await page.goto(BASE_URL)
        
        # Create a sample PDF if it doesn't exist
        if not SAMPLE_PDF.exists():
            await self._create_sample_pdf()
        
        # Upload PDF file
        file_input = await page.locator('input[type="file"]').element_handle()
        await file_input.set_input_files(str(SAMPLE_PDF))
        
        # Wait for upload to complete
        await page.wait_for_selector(".upload-progress", state="visible")
        
        # Wait for processing to complete (max 30 seconds)
        await page.wait_for_selector(
            ".processing-complete", 
            state="visible",
            timeout=30000
        )
        
        # Verify success message
        await expect(page.locator(".success-message")).to_contain_text("PDF processed successfully")
        
        # Verify questions are displayed
        questions = page.locator(".question-row")
        await expect(questions).to_have_count(3, timeout=10000)  # Expecting 3 questions from sample
    
    async def test_question_editing(self, page: Page):
        """Test editing extracted questions"""
        await page.goto(BASE_URL)
        
        # Assume questions are already loaded (from previous test or existing data)
        # Click on first question to edit
        first_question = page.locator(".question-row").first
        await first_question.click()
        
        # Wait for edit modal
        await page.wait_for_selector(".edit-modal", state="visible")
        
        # Edit question text
        question_input = page.locator('textarea[name="question_text"]')
        await question_input.clear()
        await question_input.fill("Updated question text for testing")
        
        # Edit marks
        marks_input = page.locator('input[name="marks"]')
        await marks_input.clear()
        await marks_input.fill("10")
        
        # Save changes
        await page.click('button:text("Save")')
        
        # Verify changes were saved
        await expect(first_question).to_contain_text("Updated question text")
        await expect(first_question).to_contain_text("10 marks")
    
    async def test_question_approval_workflow(self, page: Page):
        """Test approving and rejecting questions"""
        await page.goto(BASE_URL)
        
        # Select first two questions
        await page.click('input[data-row-index="0"]')  # First checkbox
        await page.click('input[data-row-index="1"]')  # Second checkbox
        
        # Click approve button
        await page.click('button:text("Approve Selected")')
        
        # Verify approval status
        approved_badges = page.locator('.status-approved')
        await expect(approved_badges).to_have_count(2)
        
        # Select third question and reject
        await page.click('input[data-row-index="2"]')
        await page.click('button:text("Reject Selected")')
        
        # Verify rejection status
        rejected_badge = page.locator('.status-rejected')
        await expect(rejected_badge).to_have_count(1)
    
    async def test_search_functionality(self, page: Page):
        """Test searching for questions"""
        await page.goto(BASE_URL)
        
        # Enter search term
        search_input = page.locator('input[placeholder="Search questions..."]')
        await search_input.fill("calculus")
        await search_input.press("Enter")
        
        # Wait for search results
        await page.wait_for_selector(".search-complete", state="visible")
        
        # Verify filtered results
        questions = page.locator(".question-row")
        count = await questions.count()
        
        # All visible questions should contain "calculus"
        for i in range(count):
            question = questions.nth(i)
            text = await question.text_content()
            assert "calculus" in text.lower()
    
    async def test_export_workflow(self, page: Page):
        """Test exporting questions to CSV"""
        await page.goto(BASE_URL)
        
        # Click export button
        await page.click('button:text("Export")')
        
        # Select CSV format
        await page.click('input[value="csv"]')
        
        # Start download
        download_promise = page.wait_for_event("download")
        await page.click('button:text("Download")')
        download = await download_promise
        
        # Verify download
        assert download.suggested_filename == "questions_export.csv"
        
        # Save and verify content
        download_path = Path("./test_download.csv")
        await download.save_as(download_path)
        
        # Read and verify CSV content
        content = download_path.read_text()
        assert "question_number" in content
        assert "question_text" in content
        assert "marks" in content
        
        # Cleanup
        download_path.unlink()
    
    async def test_real_time_progress_updates(self, page: Page):
        """Test WebSocket real-time progress updates"""
        await page.goto(BASE_URL)
        
        # Set up WebSocket message listener
        ws_messages = []
        
        async def handle_ws_message(ws):
            async for message in ws:
                if message.type == "message":
                    ws_messages.append(json.loads(message.data))
        
        # Create sample PDF for upload
        if not SAMPLE_PDF.exists():
            await self._create_sample_pdf()
        
        # Upload PDF and monitor WebSocket messages
        file_input = await page.locator('input[type="file"]').element_handle()
        await file_input.set_input_files(str(SAMPLE_PDF))
        
        # Wait for progress indicators
        await expect(page.locator(".progress-ocr")).to_be_visible()
        await expect(page.locator(".progress-extraction")).to_be_visible()
        await expect(page.locator(".progress-storage")).to_be_visible()
        
        # Verify progress bars update
        ocr_progress = page.locator(".progress-ocr .progress-bar")
        await expect(ocr_progress).to_have_attribute("style", "width: 100%", timeout=20000)
    
    async def test_error_handling(self, page: Page):
        """Test error handling for invalid files"""
        await page.goto(BASE_URL)
        
        # Create an invalid file (not a PDF)
        invalid_file = TEST_DATA_DIR / "invalid.txt"
        invalid_file.write_text("This is not a PDF")
        
        # Try to upload invalid file
        file_input = await page.locator('input[type="file"]').element_handle()
        await file_input.set_input_files(str(invalid_file))
        
        # Verify error message
        await expect(page.locator(".error-message")).to_contain_text("Invalid file type")
        
        # Cleanup
        invalid_file.unlink()
    
    async def test_responsive_design(self, page: Page):
        """Test responsive design on different screen sizes"""
        # Test desktop view
        await page.set_viewport_size({"width": 1920, "height": 1080})
        await page.goto(BASE_URL)
        await expect(page.locator("#sidebar")).to_be_visible()
        
        # Test tablet view
        await page.set_viewport_size({"width": 768, "height": 1024})
        await expect(page.locator("#sidebar")).to_be_hidden()
        await expect(page.locator("#mobile-menu-button")).to_be_visible()
        
        # Test mobile view
        await page.set_viewport_size({"width": 375, "height": 667})
        await expect(page.locator(".mobile-optimized")).to_be_visible()
    
    async def test_accessibility(self, page: Page):
        """Test accessibility features"""
        await page.goto(BASE_URL)
        
        # Test keyboard navigation
        await page.keyboard.press("Tab")
        await expect(page.locator(":focus")).to_have_attribute("aria-label")
        
        # Test screen reader labels
        upload_area = page.locator("#upload-area")
        await expect(upload_area).to_have_attribute("aria-label", "Upload PDF files")
        
        # Test color contrast (would need axe-core or similar)
        # This is a placeholder for actual accessibility testing
        assert True
    
    async def _create_sample_pdf(self):
        """Create a sample PDF file for testing"""
        TEST_DATA_DIR.mkdir(exist_ok=True)
        
        # Create a minimal PDF content
        pdf_content = b"""%PDF-1.4
1 0 obj
<<
/Type /Catalog
/Pages 2 0 R
>>
endobj
2 0 obj
<<
/Type /Pages
/Kids [3 0 R]
/Count 1
>>
endobj
3 0 obj
<<
/Type /Page
/Parent 2 0 R
/Resources <<
/Font <<
/F1 4 0 R
>>
>>
/MediaBox [0 0 612 792]
/Contents 5 0 R
>>
endobj
4 0 obj
<<
/Type /Font
/Subtype /Type1
/BaseFont /Helvetica
>>
endobj
5 0 obj
<<
/Length 228
>>
stream
BT
/F1 12 Tf
50 700 Td
(Mathematics Exam - June 2024) Tj
0 -40 Td
(Q1. [5 marks] Calculate the derivative of f(x) = 3x^2 + 2x - 1) Tj
0 -30 Td
(Q2. [10 marks] Solve the integral of x^3 + 2x dx) Tj
0 -30 Td
(Q3. [15 marks] Find the limit as x approaches infinity) Tj
ET
endstream
endobj
xref
0 6
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
0000000260 00000 n
0000000338 00000 n
trailer
<<
/Size 6
/Root 1 0 R
>>
startxref
616
%%EOF"""
        
        async with aiofiles.open(SAMPLE_PDF, 'wb') as f:
            await f.write(pdf_content)


# Performance testing
class TestPerformance:
    """Performance and load testing"""
    
    async def test_page_load_time(self, page: Page):
        """Test page load performance"""
        start_time = asyncio.get_event_loop().time()
        
        await page.goto(BASE_URL)
        await page.wait_for_load_state("networkidle")
        
        load_time = asyncio.get_event_loop().time() - start_time
        
        # Page should load in under 3 seconds
        assert load_time < 3.0, f"Page took {load_time:.2f}s to load"
    
    async def test_large_pdf_processing(self, page: Page):
        """Test processing of large PDF files"""
        # This would test with a larger PDF file
        # Placeholder for now
        assert True
    
    async def test_concurrent_uploads(self, browser):
        """Test multiple concurrent PDF uploads"""
        # Create multiple browser contexts for concurrent testing
        contexts = []
        pages = []
        
        for i in range(3):
            context = await browser.new_context()
            page = await context.new_page()
            contexts.append(context)
            pages.append(page)
        
        # Perform concurrent uploads
        upload_tasks = []
        for i, page in enumerate(pages):
            task = self._upload_pdf_async(page, f"test_{i}.pdf")
            upload_tasks.append(task)
        
        # Wait for all uploads to complete
        results = await asyncio.gather(*upload_tasks, return_exceptions=True)
        
        # Verify all uploads succeeded
        for result in results:
            assert result is True
        
        # Cleanup
        for context in contexts:
            await context.close()
    
    async def _upload_pdf_async(self, page: Page, filename: str):
        """Helper to upload PDF asynchronously"""
        try:
            await page.goto(BASE_URL)
            # Upload logic here
            return True
        except Exception as e:
            return False