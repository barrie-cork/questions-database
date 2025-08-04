"""
Manual PDF Upload Test using Playwright
This script uploads a real PDF file to the PDF Question Extractor application
"""
import asyncio
from playwright.async_api import async_playwright
import os
from pathlib import Path

async def upload_pdf_test():
    """Upload a real PDF file and monitor the processing"""
    
    # Configuration
    BASE_URL = "http://localhost:8000"
    PDF_PATH = "/mnt/d/Python/Projects/Dave/questions_pdf_to_sheet/pdf_question_extractor/tests/test_pdfs/June 2022 QP.pdf"
    
    print("🚀 Starting PDF Upload Test")
    print(f"📄 PDF File: {Path(PDF_PATH).name}")
    print(f"🌐 Target URL: {BASE_URL}")
    
    async with async_playwright() as p:
        # Launch browser in headed mode so we can see what's happening
        browser = await p.chromium.launch(
            headless=False,  # Show browser window
            slow_mo=500      # Slow down actions by 500ms for visibility
        )
        
        # Create context with viewport
        context = await browser.new_context(
            viewport={"width": 1280, "height": 800},
            locale="en-US"
        )
        
        # Create page
        page = await context.new_page()
        
        try:
            print("\n📍 Step 1: Navigating to application...")
            await page.goto(BASE_URL)
            await page.wait_for_load_state("networkidle")
            
            # Take screenshot of initial state
            await page.screenshot(path="screenshots/01_initial_state.png")
            print("✅ Application loaded")
            
            print("\n📍 Step 2: Locating upload area...")
            # Wait for upload area to be visible
            upload_area = page.locator("#upload-area")
            await upload_area.wait_for(state="visible", timeout=5000)
            print("✅ Upload area found")
            
            print("\n📍 Step 3: Uploading PDF file...")
            # Find the file input
            file_input = page.locator('input[type="file"]')
            
            # Upload the file
            await file_input.set_input_files(PDF_PATH)
            print("✅ File selected for upload")
            
            # Take screenshot after file selection
            await page.screenshot(path="screenshots/02_file_selected.png")
            
            print("\n📍 Step 4: Monitoring upload progress...")
            # Wait for upload progress to appear
            upload_progress = page.locator(".upload-progress")
            await upload_progress.wait_for(state="visible", timeout=10000)
            print("✅ Upload started")
            
            # Monitor progress stages
            print("\n📊 Processing stages:")
            
            # OCR Stage
            print("  ⏳ OCR Processing...", end="", flush=True)
            ocr_complete = page.locator(".stage-ocr.completed")
            await ocr_complete.wait_for(state="visible", timeout=60000)
            print(" ✅")
            await page.screenshot(path="screenshots/03_ocr_complete.png")
            
            # Extraction Stage
            print("  ⏳ Question Extraction...", end="", flush=True)
            extraction_complete = page.locator(".stage-extraction.completed")
            await extraction_complete.wait_for(state="visible", timeout=30000)
            print(" ✅")
            await page.screenshot(path="screenshots/04_extraction_complete.png")
            
            # Storage Stage
            print("  ⏳ Storing Questions...", end="", flush=True)
            storage_complete = page.locator(".stage-storage.completed")
            await storage_complete.wait_for(state="visible", timeout=10000)
            print(" ✅")
            await page.screenshot(path="screenshots/05_storage_complete.png")
            
            print("\n📍 Step 5: Verifying extracted questions...")
            # Wait for questions to appear in the grid
            await page.wait_for_selector(".question-row", state="visible", timeout=10000)
            
            # Count questions
            questions = page.locator(".question-row")
            question_count = await questions.count()
            print(f"✅ Found {question_count} questions")
            
            # Take final screenshot
            await page.screenshot(path="screenshots/06_questions_displayed.png", full_page=True)
            
            # Get some question details
            if question_count > 0:
                print("\n📋 Sample questions extracted:")
                for i in range(min(3, question_count)):
                    question_row = questions.nth(i)
                    question_text = await question_row.locator(".question-text").text_content()
                    marks = await question_row.locator(".marks").text_content()
                    print(f"  • Question {i+1}: {question_text[:60]}... [{marks}]")
            
            # Test search functionality
            print("\n📍 Step 6: Testing search functionality...")
            search_input = page.locator('input[placeholder*="Search"]')
            await search_input.fill("integral")
            await search_input.press("Enter")
            
            # Wait for search to complete
            await page.wait_for_timeout(2000)
            
            # Check filtered results
            filtered_questions = await questions.count()
            print(f"✅ Search returned {filtered_questions} results")
            await page.screenshot(path="screenshots/07_search_results.png")
            
            # Test question editing
            if question_count > 0:
                print("\n📍 Step 7: Testing question editing...")
                first_question = questions.first
                await first_question.click()
                
                # Wait for edit form
                edit_form = page.locator(".edit-form")
                await edit_form.wait_for(state="visible", timeout=5000)
                print("✅ Edit form opened")
                
                # Take screenshot of edit form
                await page.screenshot(path="screenshots/08_edit_form.png")
                
                # Close edit form
                close_button = page.locator('.edit-form button:has-text("Cancel")')
                if await close_button.is_visible():
                    await close_button.click()
                else:
                    await page.keyboard.press("Escape")
            
            # Test bulk selection
            print("\n📍 Step 8: Testing bulk operations...")
            # Clear search first
            await search_input.clear()
            await search_input.press("Enter")
            await page.wait_for_timeout(1000)
            
            # Select all
            select_all = page.locator('input[type="checkbox"].select-all')
            if await select_all.is_visible():
                await select_all.click()
                print("✅ Selected all questions")
                
                # Click approve button
                approve_button = page.locator('button:has-text("Approve Selected")')
                if await approve_button.is_visible():
                    await approve_button.click()
                    print("✅ Approved all questions")
                    await page.wait_for_timeout(2000)
                    await page.screenshot(path="screenshots/09_approved_questions.png")
            
            print("\n🎉 Test completed successfully!")
            print(f"📸 Screenshots saved in: screenshots/")
            
            # Keep browser open for 10 seconds to observe results
            print("\n⏰ Keeping browser open for observation...")
            await page.wait_for_timeout(10000)
            
        except Exception as e:
            print(f"\n❌ Error during test: {str(e)}")
            # Take error screenshot
            await page.screenshot(path="screenshots/error_state.png")
            raise
        
        finally:
            # Close browser
            await browser.close()

async def main():
    """Main entry point"""
    # Create screenshots directory
    os.makedirs("screenshots", exist_ok=True)
    
    # Run the test
    await upload_pdf_test()

if __name__ == "__main__":
    asyncio.run(main())