# Service Implementation Specifications

## Overview
Detailed specifications for implementing the core services using the latest APIs and best practices from the Gemini research.

## 1. OCR Service Implementation

### File: `services/ocr_service.py`

```python
from mistralai import Mistral
import base64
import logging
import os
from typing import Optional
from tenacity import retry, stop_after_attempt, wait_exponential
import aiofiles

class MistralOCRService:
    """Service for processing PDFs with Mistral OCR API
    
    Uses mistral-ocr-latest model which provides unprecedented accuracy
    in document understanding, comprehending text, tables, equations, and media.
    """
    
    def __init__(self, api_key: str):
        self.client = Mistral(api_key=api_key)
        self.logger = logging.getLogger(__name__)
        self.max_file_size = 50 * 1024 * 1024  # 50MB limit
        self.max_pages = 1000  # Maximum pages limit
        
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def process_pdf(self, pdf_path: str) -> str:
        """
        Process PDF file with Mistral OCR
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Markdown formatted text with document structure
        """
        # Validate file size
        file_size = os.path.getsize(pdf_path)
        if file_size > self.max_file_size:
            raise ValueError(f"File size {file_size} exceeds limit of {self.max_file_size}")
        
        # Read and encode PDF
        async with aiofiles.open(pdf_path, 'rb') as f:
            pdf_content = await f.read()
            pdf_base64 = base64.b64encode(pdf_content).decode('utf-8')
        
        try:
            # Process with Mistral OCR - Latest model as of 2025
            # Model options: mistral-ocr-latest, mistral-ocr-2505
            response = await self.client.ocr.process(
                model="mistral-ocr-latest",  # Best for document understanding
                document={
                    "type": "document_base64",
                    "document_base64": pdf_base64
                },
                include_image_base64=True  # Include extracted images
            )
            
            # Note: OCR preserves document structure and footnotes but not formatting
            # (bold, italics, etc). Outputs interleaved text and images.
            
            self.logger.info(f"Successfully processed {pdf_path}")
            return response.text
            
        except Exception as e:
            self.logger.error(f"OCR processing failed for {pdf_path}: {e}")
            raise
```

## 2. LLM Service Implementation

### File: `services/llm_service.py`

```python
from google import genai
from google.genai import types
import json
from typing import List, Optional
from pydantic import BaseModel, Field
from enum import Enum
from langchain_text_splitters import RecursiveCharacterTextSplitter
import asyncio
from collections import deque
import time
import logging

# Pydantic models for structured output
class QuestionType(str, Enum):
    MCQ = "MCQ"
    ESSAY = "Essay"
    SHORT_ANSWER = "Short Answer"
    NUMERICAL = "Numerical"
    TRUE_FALSE = "True/False"

class ExtractedQuestion(BaseModel):
    question_number: str = Field(..., description="Question number (e.g., '1', '2a', '3.1')")
    marks: int = Field(..., description="Marks allocated for this question", ge=0)
    question_text: str = Field(..., description="Complete question text")
    topics: List[str] = Field(..., description="Topics/subtopics covered")
    question_type: QuestionType = Field(..., description="Type of question")

class ExamPaper(BaseModel):
    year: str = Field(..., description="Year of the exam paper")
    level: str = Field(..., description="Education level (e.g., 'High School', 'A-Level')")
    subject: Optional[str] = Field(None, description="Subject name if identifiable")
    questions: List[ExtractedQuestion] = Field(..., description="All extracted questions")

# Rate limiter
class RateLimiter:
    def __init__(self, max_requests_per_minute: int = 60):
        self.max_requests = max_requests_per_minute
        self.requests = deque()
        self._lock = asyncio.Lock()
        
    async def acquire(self):
        """Wait if necessary to respect rate limits"""
        async with self._lock:
            now = time.time()
            
            # Remove old requests
            while self.requests and self.requests[0] < now - 60:
                self.requests.popleft()
            
            # If at limit, calculate wait time
            if len(self.requests) >= self.max_requests:
                oldest_request = self.requests[0]
                wait_time = 60 - (now - oldest_request) + 0.1
                await asyncio.sleep(wait_time)
                return await self.acquire()
            
            self.requests.append(now)

# Main LLM service
class GeminiLLMService:
    """Service for extracting questions using Gemini 2.5 Flash
    
    Utilizes Gemini 2.5 Flash's thinking capabilities and structured output
    for accurate question extraction with JSON schema validation.
    """
    
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)
        # Gemini 2.5 Flash - best price/performance for structured extraction
        self.model = "gemini-2.5-flash"  
        self.rate_limiter = RateLimiter(max_requests_per_minute=60)
        self.chunk_threshold = 50000  # Characters for single processing
        self.logger = logging.getLogger(__name__)
        
    async def extract_questions(self, markdown_text: str, pdf_filename: str) -> ExamPaper:
        """Extract questions from OCR markdown"""
        
        # Check if we need chunking
        if len(markdown_text) < self.chunk_threshold:
            return await self._extract_single_chunk(markdown_text, pdf_filename)
        else:
            return await self._extract_with_chunks(markdown_text, pdf_filename)
    
    async def _extract_single_chunk(self, text: str, filename: str) -> ExamPaper:
        """Extract questions from a single chunk"""
        
        await self.rate_limiter.acquire()
        
        prompt = self._create_extraction_prompt(text, filename)
        
        try:
            # Generate content with structured output using Gemini 2.5 Flash
            response = await self.client.models.generate_content_async(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=ExamPaper.model_json_schema(),
                    temperature=0.1,  # Low for consistency
                    max_output_tokens=8192,
                    top_p=0.95,
                    # Optional: Add stop_sequences, presence_penalty, frequency_penalty
                ),
                safety_settings=self._get_safety_settings()
            )
            
            # Parse response
            result = json.loads(response.text)
            exam_paper = ExamPaper.model_validate(result)
            
            # Add source filename to each question
            for question in exam_paper.questions:
                question.source_pdf = filename
                
            return exam_paper
            
        except Exception as e:
            self.logger.error(f"Question extraction failed: {e}")
            raise
    
    async def _extract_with_chunks(self, text: str, filename: str) -> ExamPaper:
        """Extract questions from large documents using chunking"""
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=40000,
            chunk_overlap=500,
            separators=["\n\n\n", "\n\n", "\n", " "],
            keep_separator=True
        )
        
        chunks = splitter.split_text(text)
        all_questions = []
        
        # Process chunks with context
        for i, chunk in enumerate(chunks):
            chunk_context = f"[Part {i+1} of {len(chunks)}]\n{chunk}"
            partial_result = await self._extract_single_chunk(chunk_context, filename)
            all_questions.extend(partial_result.questions)
        
        # Deduplicate and merge
        unique_questions = self._deduplicate_questions(all_questions)
        
        # Extract metadata from first chunk
        first_result = await self._extract_single_chunk(chunks[0], filename)
        
        return ExamPaper(
            year=first_result.year,
            level=first_result.level,
            subject=first_result.subject,
            questions=unique_questions
        )
    
    def _create_extraction_prompt(self, text: str, filename: str) -> str:
        """Create optimized extraction prompt"""
        return f"""
You are an expert at extracting questions from exam papers.

CRITICAL INSTRUCTIONS:
1. Extract EVERY question, including sub-questions (e.g., 1a, 1b, 1c)
2. Preserve the EXACT question text, including mathematical notation
3. Identify question type based on answer format expected
4. Extract marks from patterns like "[5 marks]", "(5)", "5M", "5 pts"
5. Infer topics from question content and context
6. Extract year and level from headers, footers, or content

QUESTION TYPE CLASSIFICATION:
- MCQ: Multiple choice with options (A, B, C, D)
- Essay: Requires extended written response (usually >5 marks)
- Short Answer: Brief response expected (1-5 marks)
- Numerical: Calculation or numerical answer required
- True/False: Binary choice questions

Document: {filename}
Content:
{text}

Extract all questions following the exact schema provided.
"""
    
    def _deduplicate_questions(self, questions: List[ExtractedQuestion]) -> List[ExtractedQuestion]:
        """Remove duplicate questions based on number and text"""
        seen = set()
        unique = []
        
        for q in questions:
            key = (q.question_number, q.question_text[:50])  # Use first 50 chars
            if key not in seen:
                seen.add(key)
                unique.append(q)
                
        return unique
    
    def _get_safety_settings(self):
        """Get appropriate safety settings for educational content"""
        return [
            types.SafetySetting(
                category="HARM_CATEGORY_HATE_SPEECH",
                threshold="BLOCK_ONLY_HIGH"
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_DANGEROUS_CONTENT",
                threshold="BLOCK_ONLY_HIGH"
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_HARASSMENT",
                threshold="BLOCK_ONLY_HIGH"
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                threshold="BLOCK_ONLY_HIGH"
            )
        ]
```

## 3. Embedding Service Implementation

### File: `services/embedding_service.py`

```python
from google import genai
from google.genai import types
from typing import List, Dict, Optional
import asyncio
import numpy as np
from tenacity import retry, stop_after_attempt, wait_exponential

class GeminiEmbeddingService:
    """Service for generating embeddings with Gemini"""
    
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)
        # Latest embedding model with 3072 dimensions and 8K token limit
        self.model = "models/gemini-embedding-001"  
        self.dimension = 3072  # Full dimensions (can be reduced with output_dimensionality)
        self.rate_limiter = RateLimiter(max_requests_per_minute=100)
        self.batch_size = 5  # Process 5 texts at a time
        self.logger = logging.getLogger(__name__)
        
    async def generate_embedding(
        self, 
        text: str, 
        task_type: str = "RETRIEVAL_DOCUMENT"
    ) -> List[float]:
        """Generate embedding for a single text"""
        
        await self.rate_limiter.acquire()
        
        # Configure embedding generation with task type
        config = types.EmbedContentConfig(
            task_type=task_type,  # Options: RETRIEVAL_DOCUMENT, RETRIEVAL_QUERY, etc.
            output_dimensionality=self.dimension,  # Can reduce from 3072 for storage efficiency
            # Optional: title parameter for document context
        )
        
        try:
            response = await self.client.models.embed_content_async(
                model=self.model,
                contents=text,
                config=config
            )
            
            return response.embeddings[0].values
            
        except Exception as e:
            self.logger.error(f"Embedding generation failed: {e}")
            raise
    
    async def generate_question_embeddings(
        self, 
        questions: List[Dict]
    ) -> Dict[int, List[float]]:
        """Generate embeddings for multiple questions with enrichment"""
        
        embeddings = {}
        
        # Process in batches
        for i in range(0, len(questions), self.batch_size):
            batch = questions[i:i + self.batch_size]
            
            # Create enriched texts
            enriched_texts = [
                self._create_enriched_text(q) for q in batch
            ]
            
            # Generate embeddings concurrently
            tasks = [
                self.generate_embedding(text, "RETRIEVAL_DOCUMENT")
                for text in enriched_texts
            ]
            
            batch_embeddings = await asyncio.gather(*tasks)
            
            # Store results
            for j, embedding in enumerate(batch_embeddings):
                question_id = batch[j].get('id', i + j)
                embeddings[question_id] = embedding
            
            # Small delay between batches
            await asyncio.sleep(0.1)
        
        return embeddings
    
    def _create_enriched_text(self, question: Dict) -> str:
        """Create semantically rich text for better embeddings"""
        
        # Map question types to descriptive terms
        type_descriptions = {
            "MCQ": "multiple choice question requiring selection from options",
            "Essay": "extended written response requiring detailed explanation",
            "Short Answer": "brief explanation or definition",
            "Numerical": "mathematical calculation or numerical problem",
            "True/False": "binary classification statement"
        }
        
        # Infer cognitive skill
        cognitive_skill = self._infer_cognitive_skill(question['question_text'])
        
        return f"""
Educational Assessment Question:
Question Text: {question['question_text']}
Question Number: {question['question_number']}
Answer Format: {type_descriptions.get(question['question_type'], question['question_type'])}
Subject Topics: {', '.join(question['topics'])}
Difficulty Level: {question['level']}
Assessment Year: {question['year']}
Allocated Marks: {question['marks']} marks
Cognitive Skill Required: {cognitive_skill}
Source Document: {question.get('source_pdf', 'Unknown')}
"""
    
    def _infer_cognitive_skill(self, question_text: str) -> str:
        """Infer cognitive skill level from question text"""
        text_lower = question_text.lower()
        
        # Higher-order thinking
        if any(word in text_lower for word in ['analyze', 'evaluate', 'critique', 'compare', 'justify']):
            return "Higher-order thinking - Analysis, Evaluation, and Critical Thinking"
        
        # Comprehension
        elif any(word in text_lower for word in ['explain', 'describe', 'discuss', 'elaborate', 'interpret']):
            return "Comprehension - Understanding and Explanation"
        
        # Application
        elif any(word in text_lower for word in ['calculate', 'solve', 'derive', 'compute', 'apply']):
            return "Application - Problem Solving and Computation"
        
        # Knowledge recall
        elif any(word in text_lower for word in ['define', 'list', 'state', 'identify', 'name']):
            return "Knowledge - Recall and Recognition"
        
        # Synthesis
        elif any(word in text_lower for word in ['create', 'design', 'propose', 'develop', 'formulate']):
            return "Synthesis - Creation and Innovation"
        
        else:
            return "General Cognitive Skills"
    
    async def search_embedding(
        self, 
        query: str, 
        task_type: str = "RETRIEVAL_QUERY"
    ) -> List[float]:
        """Generate embedding for search query"""
        
        # Enhance query for better search
        enhanced_query = f"Find questions about: {query}"
        
        return await self.generate_embedding(enhanced_query, task_type)
```

## 4. PDF Processor Implementation

### File: `services/pdf_processor.py`

```python
import asyncio
from pathlib import Path
from typing import Dict, List, Optional
import logging
from datetime import datetime
import hashlib
import json

# Main processor
class PDFQuestionProcessor:
    """Orchestrates the complete PDF processing pipeline"""
    
    def __init__(
        self, 
        ocr_service: MistralOCRService,
        llm_service: GeminiLLMService,
        embedding_service: GeminiEmbeddingService,
        cache_dir: Optional[Path] = None
    ):
        self.ocr = ocr_service
        self.llm = llm_service
        self.embedding = embedding_service
        self.logger = logging.getLogger(__name__)
        
        # Setup caching
        self.cache_dir = cache_dir or Path("cache")
        self.cache_dir.mkdir(exist_ok=True)
        
    async def process_pdf(
        self, 
        pdf_path: str, 
        pdf_filename: str,
        use_cache: bool = True
    ) -> Dict:
        """
        Complete pipeline for PDF processing
        
        Returns:
            Dict containing exam info and questions with embeddings
        """
        start_time = datetime.now()
        
        # Check cache
        cache_key = self._get_cache_key(pdf_path)
        if use_cache:
            cached_result = self._load_from_cache(cache_key)
            if cached_result:
                self.logger.info(f"Using cached result for {pdf_filename}")
                return cached_result
        
        try:
            # Step 1: OCR Processing
            self.logger.info(f"Starting OCR for {pdf_filename}")
            markdown_text = await self.ocr.process_pdf(pdf_path)
            
            # Log OCR completion
            self.logger.info(f"OCR completed for {pdf_filename}")
            
            # Step 2: Question Extraction
            self.logger.info("Extracting questions with LLM")
            exam_paper = await self.llm.extract_questions(markdown_text, pdf_filename)
            
            # Log extraction completion
            self.logger.info(f"Extracted {len(exam_paper.questions)} questions")
            
            # Step 3: Generate Embeddings
            self.logger.info(f"Generating embeddings for {len(exam_paper.questions)} questions")
            questions_dict = [q.model_dump() for q in exam_paper.questions]
            embeddings = await self.embedding.generate_question_embeddings(questions_dict)
            
            # Log embedding completion
            self.logger.info(f"Generated embeddings for {len(embeddings)} questions")
            
            # Combine results
            questions_with_embeddings = []
            for i, question in enumerate(exam_paper.questions):
                question_dict = question.model_dump()
                question_dict['source_pdf'] = pdf_filename
                
                questions_with_embeddings.append({
                    'question': question_dict,
                    'embedding': embeddings.get(i, [])
                })
            
            result = {
                'exam_info': {
                    'year': exam_paper.year,
                    'level': exam_paper.level,
                    'subject': exam_paper.subject,
                    'source_pdf': pdf_filename
                },
                'questions': questions_with_embeddings,
                'processing_time': (datetime.now() - start_time).total_seconds()
            }
            
            # Cache result
            if use_cache:
                self._save_to_cache(cache_key, result)
            
            self.logger.info(
                f"Completed processing {pdf_filename}: "
                f"{len(questions_with_embeddings)} questions extracted "
                f"in {result['processing_time']:.2f}s"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to process {pdf_filename}: {e}")
            raise
    
    async def process_folder(
        self, 
        folder_path: str,
        max_concurrent: int = 3
    ) -> List[Dict]:
        """Process all PDFs in a folder with concurrency control"""
        
        pdf_files = list(Path(folder_path).glob("*.pdf"))
        self.logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        # Process with limited concurrency
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_with_semaphore(pdf_path):
            async with semaphore:
                return await self.process_pdf(
                    str(pdf_path), 
                    pdf_path.name
                )
        
        results = await asyncio.gather(
            *[process_with_semaphore(pdf) for pdf in pdf_files],
            return_exceptions=True
        )
        
        # Filter out errors
        successful_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"Failed to process {pdf_files[i]}: {result}")
            else:
                successful_results.append(result)
        
        self.logger.info(
            f"Processed {len(successful_results)}/{len(pdf_files)} PDFs successfully"
        )
        
        return successful_results
    
    def _get_cache_key(self, pdf_path: str) -> str:
        """Generate cache key from file content"""
        with open(pdf_path, 'rb') as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        return f"pdf_{file_hash}"
    
    def _load_from_cache(self, cache_key: str) -> Optional[Dict]:
        """Load result from cache if exists"""
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            with open(cache_file, 'r') as f:
                return json.load(f)
        return None
    
    def _save_to_cache(self, cache_key: str, result: Dict):
        """Save result to cache"""
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        # Don't cache embeddings (too large)
        result_to_cache = result.copy()
        for item in result_to_cache['questions']:
            item['embedding'] = None
        
        with open(cache_file, 'w') as f:
            json.dump(result_to_cache, f)
```

## 5. Performance Monitoring

### File: `services/monitoring.py`

```python
import time
from datetime import datetime
from typing import Dict, List
import logging
from prometheus_client import Counter, Histogram

# Prometheus metrics
pdf_processed_total = Counter('pdf_processed_total', 'Total PDFs processed')
questions_extracted_total = Counter('questions_extracted_total', 'Total questions extracted')
processing_duration = Histogram('processing_duration_seconds', 'PDF processing duration')
api_errors_total = Counter('api_errors_total', 'Total API errors', ['service'])

class PerformanceMonitor:
    """Monitor and track performance metrics"""
    
    def __init__(self):
        self.metrics = {
            'pdfs_processed': 0,
            'questions_extracted': 0,
            'total_processing_time': 0,
            'errors': {'ocr': 0, 'llm': 0, 'embedding': 0},
            'api_calls': {'ocr': 0, 'llm': 0, 'embedding': 0}
        }
        self.logger = logging.getLogger(__name__)
    
    def track_pdf_processing(self, questions_count: int, duration: float):
        """Track successful PDF processing"""
        self.metrics['pdfs_processed'] += 1
        self.metrics['questions_extracted'] += questions_count
        self.metrics['total_processing_time'] += duration
        
        # Update Prometheus metrics
        pdf_processed_total.inc()
        questions_extracted_total.inc(questions_count)
        processing_duration.observe(duration)
    
    def track_error(self, service: str):
        """Track API errors"""
        self.metrics['errors'][service] += 1
        api_errors_total.labels(service=service).inc()
    
    def track_api_call(self, service: str):
        """Track API calls"""
        self.metrics['api_calls'][service] += 1
    
    def get_summary(self) -> Dict:
        """Get performance summary"""
        avg_time = (
            self.metrics['total_processing_time'] / self.metrics['pdfs_processed']
            if self.metrics['pdfs_processed'] > 0 else 0
        )
        
        avg_questions = (
            self.metrics['questions_extracted'] / self.metrics['pdfs_processed']
            if self.metrics['pdfs_processed'] > 0 else 0
        )
        
        return {
            'total_pdfs': self.metrics['pdfs_processed'],
            'total_questions': self.metrics['questions_extracted'],
            'average_processing_time': avg_time,
            'average_questions_per_pdf': avg_questions,
            'error_rates': {
                service: errors / self.metrics['api_calls'].get(service, 1)
                for service, errors in self.metrics['errors'].items()
            },
            'api_calls': self.metrics['api_calls']
        }
```

## Key Implementation Notes

1. **Rate Limiting**: Both Gemini services include rate limiting to prevent 429 errors
2. **Retry Logic**: All external API calls have exponential backoff retry
3. **Caching**: OCR results are cached to avoid reprocessing
4. **Batch Processing**: Embeddings are generated in batches for efficiency
5. **Enhanced Embeddings**: Questions are enriched with metadata for better search
6. **Structured Output**: Using Pydantic models with Gemini's native JSON schema support
7. **Error Handling**: Comprehensive error handling and logging throughout
8. **Performance Metrics**: Prometheus metrics for monitoring in production
9. **Smart Chunking**: Large documents are intelligently split with context preservation

## API Updates (2025)

### Mistral OCR
- **Model**: `mistral-ocr-latest` (or `mistral-ocr-2505` for specific version)
- **Capabilities**: Comprehends text, tables, equations, media with unprecedented accuracy
- **Limits**: 50MB file size, 1000 pages maximum
- **Output**: Interleaved text and images in Markdown format
- **Pricing**: 1000 pages/$1 (double with batch inference)

### Gemini 2.5 Flash
- **Model**: `gemini-2.5-flash` (stable version as of 2025)
- **Features**: Native structured JSON output with schema validation
- **Performance**: 22% efficiency gains, best price/performance ratio
- **Context**: 1M token context window for processing entire documents

### Gemini Embeddings
- **Model**: `gemini-embedding-001` (3072 dimensions, 8K token limit)
- **Task Types**: RETRIEVAL_DOCUMENT, RETRIEVAL_QUERY, SEMANTIC_SIMILARITY, etc.
- **Features**: Matryoshka Representation Learning for dimension reduction
- **Performance**: MTEB score of 68.32, leading multilingual performance