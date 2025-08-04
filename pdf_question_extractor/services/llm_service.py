import asyncio
import logging
from typing import List, Dict, Any, Optional
from enum import Enum
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import httpx

logger = logging.getLogger(__name__)

# Pydantic models for structured output
class QuestionType(str, Enum):
    MCQ = "MCQ"
    ESSAY = "Essay"
    SHORT_ANSWER = "Short Answer"
    NUMERICAL = "Numerical"
    TRUE_FALSE = "True/False"
    FILL_IN_BLANKS = "Fill in the Blanks"
    MATCHING = "Matching"
    DIAGRAM = "Diagram/Drawing"

class Question(BaseModel):
    question_number: str = Field(..., description="Question number (e.g., '1', '2a', '3.1')")
    marks: int = Field(..., description="Marks allocated for this question")
    question_text: str = Field(..., description="Complete question text including all sub-parts")
    topics: List[str] = Field(..., description="List of topics/subtopics covered")
    question_type: QuestionType = Field(..., description="Type of question")

class ExamPaper(BaseModel):
    year: str = Field(..., description="Year of examination (e.g., '2024', '2023-24')")
    level: str = Field(..., description="Education level (e.g., 'High School', 'Grade 10', 'A-Level')")
    subject: str = Field(..., description="Subject name")
    total_marks: int = Field(..., description="Total marks for the paper")
    duration: Optional[str] = Field(None, description="Exam duration if mentioned")
    source_pdf: str = Field(..., description="Source PDF filename")
    questions: List[Question] = Field(..., description="List of extracted questions")

# Import shared RateLimiter
from .utils import RateLimiter

class GeminiLLMService:
    """Service for question extraction using Gemini 2.5 Flash"""
    
    def __init__(self, api_key: str):
        """Initialize Gemini client"""
        self.client = genai.Client(api_key=api_key)
        self.model = "gemini-2.5-flash"
        self.rate_limiter = RateLimiter(calls_per_minute=60)
        
        # Text splitter for large documents
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=50000,  # Characters per chunk
            chunk_overlap=200,  # Overlap to maintain context
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def _create_extraction_prompt(self, text: str, pdf_filename: str) -> str:
        """Create prompt for question extraction"""
        return f"""Extract all exam questions from the following document. 
        
The document is from a PDF file named: {pdf_filename}

Analyze the text carefully and extract:
1. Every question with its complete text (including all sub-parts)
2. The marks allocated for each question
3. The question type (MCQ, Essay, Short Answer, etc.)
4. Topics or subtopics the question covers
5. The year, level, and subject of the exam

Important instructions:
- Include ALL questions, even if they seem incomplete
- For multi-part questions (e.g., 1a, 1b), extract each part separately
- Preserve the exact question numbering from the source
- If marks are shown as [5] or (5 marks) or similar, extract the number
- For MCQ questions, include all options in the question text
- If a question references a diagram or image, mention it in the question text

Document text:
{text}"""
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException))
    )
    async def extract_questions(self, ocr_text: str, pdf_filename: str) -> ExamPaper:
        """
        Extract questions from OCR text using Gemini
        
        Args:
            ocr_text: Markdown text from OCR
            pdf_filename: Name of source PDF file
            
        Returns:
            ExamPaper object with extracted questions
        """
        try:
            # Rate limiting
            await self.rate_limiter.acquire()
            
            # Check if text needs chunking
            if len(ocr_text) > 50000:
                logger.info(f"Text too long ({len(ocr_text)} chars), using chunking strategy")
                return await self._extract_with_chunking(ocr_text, pdf_filename)
            
            # Single pass extraction for smaller documents
            prompt = self._create_extraction_prompt(ocr_text, pdf_filename)
            
            logger.info(f"Extracting questions from {pdf_filename} ({len(ocr_text)} chars)")
            
            response = await self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=ExamPaper.model_json_schema(),
                    temperature=0.1,  # Low temperature for consistency
                    max_output_tokens=8192
                )
            )
            
            # Parse structured response
            exam_paper = ExamPaper.model_validate_json(response.text)
            
            # Ensure source_pdf is set correctly
            exam_paper.source_pdf = pdf_filename
            
            logger.info(f"Extracted {len(exam_paper.questions)} questions from {pdf_filename}")
            
            return exam_paper
            
        except Exception as e:
            logger.error(f"Question extraction failed for {pdf_filename}: {str(e)}")
            raise
    
    async def _extract_with_chunking(self, ocr_text: str, pdf_filename: str) -> ExamPaper:
        """
        Extract questions from large documents using chunking
        
        Args:
            ocr_text: Long markdown text from OCR
            pdf_filename: Name of source PDF file
            
        Returns:
            ExamPaper object with extracted questions
        """
        # Split text into chunks
        chunks = self.text_splitter.split_text(ocr_text)
        logger.info(f"Split document into {len(chunks)} chunks")
        
        all_questions = []
        exam_info = None
        
        # Process each chunk
        for i, chunk in enumerate(chunks):
            logger.info(f"Processing chunk {i+1}/{len(chunks)}")
            
            # Extract from chunk
            chunk_result = await self.extract_questions(chunk, f"{pdf_filename}_chunk_{i}")
            
            # Collect questions
            all_questions.extend(chunk_result.questions)
            
            # Keep exam info from first chunk
            if i == 0:
                exam_info = chunk_result
        
        # Combine results
        if exam_info:
            exam_info.questions = all_questions
            exam_info.source_pdf = pdf_filename
            return exam_info
        else:
            # Fallback if no exam info extracted
            return ExamPaper(
                year="Unknown",
                level="Unknown",
                subject="Unknown",
                total_marks=sum(q.marks for q in all_questions),
                source_pdf=pdf_filename,
                questions=all_questions
            )
    
    async def enhance_question(self, question: Question) -> Question:
        """
        Enhance a single question with additional metadata or corrections
        
        Args:
            question: Question object to enhance
            
        Returns:
            Enhanced Question object
        """
        # This could be used to:
        # - Validate question completeness
        # - Add cognitive skill tags
        # - Correct OCR errors
        # - Standardize formatting
        
        return question