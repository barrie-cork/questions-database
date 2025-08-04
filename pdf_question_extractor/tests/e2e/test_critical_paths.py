"""
Critical Path E2E Tests
Tests the most important user journeys
"""
import pytest
import asyncio
from pathlib import Path
import os
from playwright.async_api import Page, expect
import aiofiles

BASE_URL = os.getenv("TEST_BASE_URL", "http://localhost:8000")


class TestCriticalPaths:
    """Test critical user paths that must always work"""
    
    @pytest.mark.critical
    async def test_complete_workflow_pdf_to_export(self, page: Page):
        """Test the complete workflow from PDF upload to export"""
        # 1. Navigate to application
        await page.goto(BASE_URL)
        await page.wait_for_load_state("networkidle")
        
        # 2. Verify initial state
        await expect(page.locator("h1")).to_contain_text("PDF Question Extractor")
        questions_grid = page.locator("#questions-grid")
        await expect(questions_grid).to_be_visible()
        
        # 3. Create and upload test PDF
        test_pdf = Path("./test_exam.pdf")
        await self._create_test_pdf(test_pdf)
        
        # Upload the PDF
        await page.set_input_files('input[type="file"]', str(test_pdf))
        
        # 4. Monitor upload progress
        upload_progress = page.locator(".upload-progress")
        await expect(upload_progress).to_be_visible(timeout=5000)
        
        # Wait for processing stages
        await expect(page.locator(".stage-ocr.completed")).to_be_visible(timeout=30000)
        await expect(page.locator(".stage-extraction.completed")).to_be_visible(timeout=30000)
        await expect(page.locator(".stage-storage.completed")).to_be_visible(timeout=30000)
        
        # 5. Verify questions loaded
        await page.wait_for_selector(".question-row", state="visible", timeout=10000)
        questions = page.locator(".question-row")
        question_count = await questions.count()
        assert question_count >= 3, f"Expected at least 3 questions, got {question_count}"
        
        # 6. Edit first question
        first_question = questions.first
        await first_question.click()
        
        # Wait for edit form
        edit_form = page.locator(".edit-form")
        await expect(edit_form).to_be_visible()
        
        # Update marks
        marks_input = page.locator('input[name="marks"]')
        await marks_input.clear()
        await marks_input.fill("20")
        
        # Save
        await page.click('button:text("Save")')
        await expect(edit_form).not_to_be_visible()
        
        # 7. Approve all questions
        select_all = page.locator('input[type="checkbox"].select-all')
        await select_all.click()
        
        await page.click('button:text("Approve Selected")')
        
        # Verify all approved
        approved_badges = page.locator('.status-badge.approved')
        await expect(approved_badges).to_have_count(question_count)
        
        # 8. Export questions
        await page.click('button:text("Export")')
        
        # Wait for export dialog
        export_dialog = page.locator(".export-dialog")
        await expect(export_dialog).to_be_visible()
        
        # Select CSV format
        await page.click('label:text("CSV Format")')
        
        # Download
        download_promise = page.wait_for_event("download")
        await page.click('button:text("Download")')
        download = await download_promise
        
        # 9. Verify download
        assert "questions" in download.suggested_filename
        assert download.suggested_filename.endswith(".csv")
        
        # Cleanup
        test_pdf.unlink(missing_ok=True)
        
        # Success!
        print("✅ Complete workflow test passed!")
    
    @pytest.mark.critical
    async def test_error_recovery_workflow(self, page: Page):
        """Test that users can recover from errors"""
        await page.goto(BASE_URL)
        
        # 1. Try uploading non-PDF file
        invalid_file = Path("./test.txt")
        invalid_file.write_text("Not a PDF")
        
        await page.set_input_files('input[type="file"]', str(invalid_file))
        
        # Verify error message
        error_message = page.locator(".error-message")
        await expect(error_message).to_be_visible()
        await expect(error_message).to_contain_text("must be a PDF")
        
        # 2. Dismiss error
        dismiss_button = page.locator('.error-message button:text("Dismiss")')
        await dismiss_button.click()
        await expect(error_message).not_to_be_visible()
        
        # 3. Upload valid PDF
        valid_pdf = Path("./valid_test.pdf")
        await self._create_test_pdf(valid_pdf)
        
        await page.set_input_files('input[type="file"]', str(valid_pdf))
        
        # Verify upload proceeds
        await expect(page.locator(".upload-progress")).to_be_visible()
        
        # Cleanup
        invalid_file.unlink(missing_ok=True)
        valid_pdf.unlink(missing_ok=True)
    
    @pytest.mark.critical
    async def test_concurrent_user_actions(self, page: Page):
        """Test that the app handles concurrent actions correctly"""
        await page.goto(BASE_URL)
        
        # Simulate rapid user actions
        # 1. Start a search
        search_input = page.locator('input[placeholder*="Search"]')
        await search_input.fill("mathematics")
        search_promise = search_input.press("Enter")
        
        # 2. Immediately click on a filter
        filter_promise = page.click('button:text("Filter")')
        
        # 3. Try to export at the same time
        export_promise = page.click('button:text("Export")')
        
        # Wait for all actions
        await asyncio.gather(
            search_promise,
            filter_promise,
            export_promise,
            return_exceptions=True
        )
        
        # App should still be responsive
        await expect(page.locator("#app")).to_be_visible()
        
        # No error messages should appear
        error_messages = page.locator(".error-message")
        error_count = await error_messages.count()
        assert error_count == 0, "Concurrent actions caused errors"
    
    @pytest.mark.critical
    async def test_data_persistence(self, page: Page):
        """Test that data persists across page refreshes"""
        await page.goto(BASE_URL)
        
        # Get initial question count
        await page.wait_for_selector(".question-row", state="visible", timeout=5000)
        initial_questions = page.locator(".question-row")
        initial_count = await initial_questions.count()
        
        # Edit a question
        if initial_count > 0:
            first_question = initial_questions.first
            question_text = await first_question.text_content()
            
            # Refresh page
            await page.reload()
            await page.wait_for_load_state("networkidle")
            
            # Verify questions still there
            await page.wait_for_selector(".question-row", state="visible")
            refreshed_questions = page.locator(".question-row")
            refreshed_count = await refreshed_questions.count()
            
            assert refreshed_count == initial_count, "Questions lost after refresh"
            
            # Verify content unchanged
            first_after_refresh = refreshed_questions.first
            text_after_refresh = await first_after_refresh.text_content()
            assert text_after_refresh == question_text, "Question content changed after refresh"
    
    @pytest.mark.critical
    async def test_api_health_monitoring(self, page: Page):
        """Test that health endpoint is accessible"""
        # Direct API call
        response = await page.request.get(f"{BASE_URL}/health")
        assert response.ok, "Health endpoint not responding"
        
        health_data = await response.json()
        assert health_data["status"] == "healthy"
        assert health_data["database_connected"] is True
        
        # Check all services
        services = health_data.get("services", {})
        for service_name, status in services.items():
            assert status in ["healthy", "configured"], f"{service_name} is not healthy"
    
    async def _create_test_pdf(self, filepath: Path):
        """Create a test PDF file"""
        pdf_content = b"""%PDF-1.4
1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj
2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj
3 0 obj<</Type/Page/Parent 2 0 R/Resources<</Font<</F1 4 0 R>>>>/MediaBox[0 0 612 792]/Contents 5 0 R>>endobj
4 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj
5 0 obj<</Length 300>>stream
BT /F1 12 Tf 50 750 Td (Mathematics Exam 2024) Tj 0 -30 Td 
(Q1. [5 marks] What is the derivative of x^2?) Tj 0 -30 Td
(Q2. [10 marks] Solve the integral of sin(x)dx) Tj 0 -30 Td
(Q3. [15 marks] Find the eigenvalues of matrix A) Tj ET
endstream
endobj
xref
0 6
0000000000 65535 f
0000000009 00000 n
0000000052 00000 n
0000000101 00000 n
0000000238 00000 n
0000000311 00000 n
trailer<</Size 6/Root 1 0 R>>startxref
663
%%EOF"""
        
        async with aiofiles.open(filepath, 'wb') as f:
            await f.write(pdf_content)


class TestUserInterface:
    """Test UI-specific functionality"""
    
    async def test_drag_and_drop_upload(self, page: Page):
        """Test drag and drop file upload"""
        await page.goto(BASE_URL)
        
        # Get the drop zone
        drop_zone = page.locator("#upload-area")
        await expect(drop_zone).to_be_visible()
        
        # Simulate drag over
        await drop_zone.dispatch_event("dragenter")
        
        # Verify visual feedback
        await expect(drop_zone).to_have_class(/drag-over/)
        
        # Simulate drag leave
        await drop_zone.dispatch_event("dragleave")
        await expect(drop_zone).not_to_have_class(/drag-over/)
    
    async def test_table_sorting(self, page: Page):
        """Test question table sorting functionality"""
        await page.goto(BASE_URL)
        
        # Wait for questions to load
        await page.wait_for_selector(".question-row", state="visible")
        
        # Click on marks column header to sort
        marks_header = page.locator('th:text("Marks")')
        await marks_header.click()
        
        # Verify sort indicator appears
        await expect(marks_header.locator(".sort-asc")).to_be_visible()
        
        # Click again for descending sort
        await marks_header.click()
        await expect(marks_header.locator(".sort-desc")).to_be_visible()
    
    async def test_pagination(self, page: Page):
        """Test pagination controls"""
        await page.goto(BASE_URL)
        
        # Check if pagination exists (only if enough questions)
        pagination = page.locator(".pagination")
        if await pagination.is_visible():
            # Click next page
            next_button = page.locator('.pagination button:text("Next")')
            await next_button.click()
            
            # Verify page changed
            current_page = page.locator(".pagination .current-page")
            await expect(current_page).to_contain_text("2")
            
            # Go back
            prev_button = page.locator('.pagination button:text("Previous")')
            await prev_button.click()
            await expect(current_page).to_contain_text("1")
    
    async def test_keyboard_shortcuts(self, page: Page):
        """Test keyboard shortcuts"""
        await page.goto(BASE_URL)
        
        # Ctrl+A to select all
        await page.keyboard.press("Control+A")
        
        # Verify all checkboxes selected
        checkboxes = page.locator('input[type="checkbox"].question-select:checked')
        all_checkboxes = page.locator('input[type="checkbox"].question-select')
        
        checked_count = await checkboxes.count()
        total_count = await all_checkboxes.count()
        
        if total_count > 0:
            assert checked_count == total_count, "Ctrl+A didn't select all"
        
        # Escape to deselect
        await page.keyboard.press("Escape")
        checked_after = await checkboxes.count()
        assert checked_after == 0, "Escape didn't deselect all"