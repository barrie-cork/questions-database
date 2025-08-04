# Gemini API Implementation Guide for google-genai SDK (August 2025)

## Overview

This guide provides up-to-date implementation details for using the **google-genai** SDK (version 0.9.0) with Gemini models in August 2025. The google-genai SDK is the new, recommended library that has replaced the deprecated google-generativeai package.

## Important Migration Notice

⚠️ **The google-generativeai package is deprecated**. All support for the old repository will permanently end on September 30, 2025. This project uses the new `google-genai` SDK.

## Table of Contents
1. [Installation and Setup](#installation-and-setup)
2. [Client Configuration](#client-configuration)
3. [Structured Output with Gemini 2.5 Flash Lite](#structured-output-with-gemini-25-flash-lite)
4. [Embeddings with gemini-embedding-001](#embeddings-with-gemini-embedding-001)
5. [Complete Implementation Examples](#complete-implementation-examples)
6. [Best Practices](#best-practices)

---

## Installation and Setup

```bash
# Basic installation
pip install google-genai==0.9.0

# With async performance optimization
pip install google-genai[aiohttp]==0.9.0

# Additional dependencies for our project
pip install langchain-text-splitters==0.3.4  # For smart chunking
pip install pydantic==2.10.4  # For structured output schemas
```

## Client Configuration

### Basic Client Setup

```python
from google import genai
import os

# Option 1: Using API key directly
client = genai.Client(api_key=os.getenv('GOOGLE_API_KEY'))

# Option 2: Auto-detect from environment
# Set GOOGLE_API_KEY or GEMINI_API_KEY environment variable
client = genai.Client()

# Option 3: For Vertex AI (if needed)
client = genai.Client(
    vertexai=True,
    project='your-project-id',
    location='us-central1'
)
```

### Async Client Setup

```python
import asyncio
from google import genai

async def create_async_client():
    # Async client for better performance
    client = genai.Client(
        api_key=os.getenv('GOOGLE_API_KEY'),
        http_client=True  # Uses aiohttp if installed
    )
    return client
```

---

## Structured Output with Gemini 2.5 Flash Lite

### Define Pydantic Models

```python
from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum

class QuestionType(str, Enum):
    MCQ = "MCQ"
    ESSAY = "Essay"
    SHORT_ANSWER = "Short Answer"
    NUMERICAL = "Numerical"
    TRUE_FALSE = "True/False"

class ExtractedQuestion(BaseModel):
    question_number: str = Field(..., description="Question number (e.g., '1', '2a', '3.1')")
    marks: int = Field(..., description="Marks allocated for this question")
    question_text: str = Field(..., description="Complete question text")
    topics: List[str] = Field(..., description="Topics/subtopics covered")
    question_type: QuestionType = Field(..., description="Type of question")

class ExamPaper(BaseModel):
    year: str = Field(..., description="Year of the exam paper")
    level: str = Field(..., description="Education level (e.g., 'High School', 'A-Level')")
    subject: Optional[str] = Field(None, description="Subject name if identifiable")
    questions: List[ExtractedQuestion] = Field(..., description="All extracted questions")
```

### Implement Question Extraction

```python
from google import genai
from google.genai import types
import json

class GeminiLLMService:
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)
        self.model = "gemini-2.5-flash-lite-001"
        
    async def extract_questions(self, markdown_text: str, pdf_filename: str) -> ExamPaper:
        """Extract questions using structured output"""
        
        prompt = f"""
        You are an expert at extracting questions from exam papers.
        
        Extract ALL questions from the following exam paper content, including:
        - Main questions and all sub-questions (1a, 1b, etc.)
        - Marks for each question
        - Question type based on expected answer format
        - Topics covered
        - Year and education level
        
        Content from {pdf_filename}:
        {markdown_text}
        """
        
        try:
            response = await self.client.models.generate_content_async(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=ExamPaper.model_json_schema(),
                    temperature=0.1,  # Low temperature for consistency
                    max_output_tokens=8192,
                    top_p=0.95
                )
            )
            
            # Parse the structured response
            result = json.loads(response.text)
            exam_paper = ExamPaper.model_validate(result)
            
            # Add source filename
            for question in exam_paper.questions:
                question.source_pdf = pdf_filename
                
            return exam_paper
            
        except Exception as e:
            print(f"Error extracting questions: {e}")
            raise
```

### Smart Chunking for Large Documents

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

class SmartDocumentProcessor:
    def __init__(self, llm_service: GeminiLLMService):
        self.llm = llm_service
        self.chunk_threshold = 50000  # Characters
        
    async def process_document(self, markdown_text: str, filename: str) -> ExamPaper:
        """Process document with intelligent chunking"""
        
        if len(markdown_text) < self.chunk_threshold:
            # Process in single call
            return await self.llm.extract_questions(markdown_text, filename)
        
        # Use smart chunking for large documents
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=40000,
            chunk_overlap=500,
            separators=["\n\n\n", "\n\n", "\n", " "],
            keep_separator=True
        )
        
        chunks = splitter.split_text(markdown_text)
        all_questions = []
        
        # Process chunks with context
        for i, chunk in enumerate(chunks):
            chunk_context = f"[Part {i+1} of {len(chunks)}]\n{chunk}"
            partial_result = await self.llm.extract_questions(chunk_context, filename)
            all_questions.extend(partial_result.questions)
        
        # Merge and deduplicate
        return self._merge_results(all_questions, filename)
```

---

## Embeddings with gemini-embedding-001

### Basic Embedding Service

```python
from google import genai
from google.genai import types
from typing import List, Dict, Optional
import numpy as np

class GeminiEmbeddingService:
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)
        self.model = "models/embedding-001"  # Note: use "models/" prefix
        self.dimension = 768
        
    async def generate_embedding(
        self, 
        text: str, 
        task_type: Optional[str] = "RETRIEVAL_DOCUMENT"
    ) -> List[float]:
        """Generate embedding for a single text"""
        
        config = types.EmbedContentConfig(
            task_type=task_type,
            output_dimensionality=self.dimension
        )
        
        response = await self.client.models.embed_content_async(
            model=self.model,
            contents=text,
            config=config
        )
        
        # Extract embedding values
        return response.embeddings[0].values
    
    async def generate_batch_embeddings(
        self, 
        texts: List[str],
        task_type: Optional[str] = "RETRIEVAL_DOCUMENT"
    ) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        
        config = types.EmbedContentConfig(
            task_type=task_type,
            output_dimensionality=self.dimension
        )
        
        # Process in batches to avoid rate limits
        embeddings = []
        batch_size = 5  # Adjust based on rate limits
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            # Generate embeddings for batch
            tasks = []
            for text in batch:
                task = self.client.models.embed_content_async(
                    model=self.model,
                    contents=text,
                    config=config
                )
                tasks.append(task)
            
            # Wait for all embeddings in batch
            batch_responses = await asyncio.gather(*tasks)
            
            # Extract embeddings
            for response in batch_responses:
                embeddings.append(response.embeddings[0].values)
            
            # Small delay to respect rate limits
            await asyncio.sleep(0.1)
        
        return embeddings
```

### Enhanced Embedding Generation

```python
class EnhancedEmbeddingService(GeminiEmbeddingService):
    """Enhanced service with semantic enrichment"""
    
    async def generate_question_embedding(self, question: Dict) -> List[float]:
        """Generate semantically rich embedding for a question"""
        
        # Create enriched representation
        enriched_text = self._create_enriched_text(question)
        
        # Generate embedding with appropriate task type
        return await self.generate_embedding(
            text=enriched_text,
            task_type="RETRIEVAL_DOCUMENT"
        )
    
    def _create_enriched_text(self, question: Dict) -> str:
        """Create semantically rich text for better embeddings"""
        
        # Map question types to descriptive terms
        type_descriptions = {
            "MCQ": "multiple choice question",
            "Essay": "extended written response",
            "Short Answer": "brief explanation",
            "Numerical": "mathematical calculation",
            "True/False": "binary classification"
        }
        
        return f"""
        Educational Assessment Question:
        Question Text: {question['question_text']}
        Question Number: {question['question_number']}
        Answer Type: {type_descriptions.get(question['question_type'], question['question_type'])}
        Subject Topics: {', '.join(question['topics'])}
        Difficulty Level: {question['level']}
        Assessment Year: {question['year']}
        Allocated Marks: {question['marks']} marks
        Cognitive Skill: {self._infer_cognitive_skill(question['question_text'])}
        """
    
    def _infer_cognitive_skill(self, question_text: str) -> str:
        """Infer cognitive skill level from question text"""
        text_lower = question_text.lower()
        
        if any(word in text_lower for word in ['analyze', 'evaluate', 'critique', 'compare']):
            return "Higher-order thinking - Analysis and Evaluation"
        elif any(word in text_lower for word in ['explain', 'describe', 'discuss', 'elaborate']):
            return "Comprehension and Understanding"
        elif any(word in text_lower for word in ['calculate', 'solve', 'derive', 'compute']):
            return "Application and Problem Solving"
        elif any(word in text_lower for word in ['define', 'list', 'state', 'identify']):
            return "Knowledge Recall and Recognition"
        else:
            return "General Cognitive Skills"
```

---

## Complete Implementation Examples

### 1. Full Processing Pipeline

```python
# services/ocr_service.py
from mistralai import Mistral
import base64

class MistralOCRService:
    def __init__(self, api_key: str):
        self.client = Mistral(api_key=api_key)
        
    async def process_pdf(self, pdf_path: str) -> str:
        """Process PDF with Mistral OCR"""
        with open(pdf_path, 'rb') as f:
            pdf_base64 = base64.b64encode(f.read()).decode()
        
        response = await self.client.ocr.process_async(
            model="mistral-ocr-latest",
            document={
                "type": "document_base64",
                "document_base64": pdf_base64
            },
            include_image_base64=True
        )
        
        return response.text

# services/pdf_processor.py
class PDFQuestionProcessor:
    def __init__(self, ocr_service, llm_service, embedding_service):
        self.ocr = ocr_service
        self.llm = llm_service
        self.embedding = embedding_service
        
    async def process_pdf(self, pdf_path: str, pdf_filename: str):
        """Complete pipeline for PDF processing"""
        
        # Step 1: OCR
        print(f"Processing {pdf_filename} with OCR...")
        markdown_text = await self.ocr.process_pdf(pdf_path)
        
        # Step 2: Extract questions
        print("Extracting questions...")
        processor = SmartDocumentProcessor(self.llm)
        exam_paper = await processor.process_document(markdown_text, pdf_filename)
        
        # Step 3: Generate embeddings
        print("Generating embeddings...")
        questions_with_embeddings = []
        
        for question in exam_paper.questions:
            question_dict = question.model_dump()
            embedding = await self.embedding.generate_question_embedding(question_dict)
            
            questions_with_embeddings.append({
                'question': question_dict,
                'embedding': embedding
            })
        
        return {
            'exam_info': {
                'year': exam_paper.year,
                'level': exam_paper.level,
                'subject': exam_paper.subject
            },
            'questions': questions_with_embeddings
        }
```

### 2. Database Storage with pgvector

```python
# database/vector_operations.py
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
import json

class VectorDatabaseOperations:
    def __init__(self, db_session: AsyncSession):
        self.session = db_session
        
    async def store_question_with_embedding(
        self, 
        question: Dict, 
        embedding: List[float]
    ) -> int:
        """Store question and embedding atomically"""
        
        async with self.session.begin():
            # Insert question
            result = await self.session.execute(
                text("""
                    INSERT INTO questions (
                        question_number, marks, year, level, topics,
                        question_type, question_text, source_pdf
                    ) VALUES (
                        :q_num, :marks, :year, :level, :topics::text[],
                        :q_type, :q_text, :source
                    ) RETURNING id
                """),
                {
                    'q_num': question['question_number'],
                    'marks': question['marks'],
                    'year': question['year'],
                    'level': question['level'],
                    'topics': question['topics'],
                    'q_type': question['question_type'],
                    'q_text': question['question_text'],
                    'source': question['source_pdf']
                }
            )
            
            question_id = result.scalar()
            
            # Store embedding
            await self.session.execute(
                text("""
                    INSERT INTO question_embeddings (
                        question_id, embedding, model_name, model_version
                    ) VALUES (
                        :q_id, :embedding::vector(768), :model, :version
                    )
                """),
                {
                    'q_id': question_id,
                    'embedding': json.dumps(embedding),
                    'model': 'gemini-embedding-001',
                    'version': '1.0'
                }
            )
            
        return question_id
    
    async def find_similar_questions(
        self, 
        query_embedding: List[float], 
        threshold: float = 0.8,
        limit: int = 10
    ) -> List[Dict]:
        """Find similar questions using cosine similarity"""
        
        result = await self.session.execute(
            text("""
                SELECT 
                    q.*,
                    1 - (qe.embedding <=> :query_embedding::vector(768)) AS similarity
                FROM questions q
                JOIN question_embeddings qe ON q.id = qe.question_id
                WHERE 1 - (qe.embedding <=> :query_embedding::vector(768)) > :threshold
                ORDER BY qe.embedding <=> :query_embedding::vector(768)
                LIMIT :limit
            """),
            {
                'query_embedding': json.dumps(query_embedding),
                'threshold': threshold,
                'limit': limit
            }
        )
        
        return [dict(row) for row in result]
```

### 3. FastAPI Endpoints

```python
# api/routes/questions.py
from fastapi import APIRouter, HTTPException, Depends
from typing import List
import asyncio

router = APIRouter(prefix="/api/questions", tags=["questions"])

@router.post("/extract")
async def extract_questions_from_pdf(
    pdf_file: UploadFile,
    processor: PDFQuestionProcessor = Depends(get_processor),
    db: AsyncSession = Depends(get_db)
):
    """Extract questions from uploaded PDF"""
    
    # Save uploaded file
    pdf_path = f"uploads/{pdf_file.filename}"
    async with aiofiles.open(pdf_path, 'wb') as f:
        content = await pdf_file.read()
        await f.write(content)
    
    try:
        # Process PDF
        result = await processor.process_pdf(pdf_path, pdf_file.filename)
        
        # Store in database
        db_ops = VectorDatabaseOperations(db)
        stored_questions = []
        
        for item in result['questions']:
            question_id = await db_ops.store_question_with_embedding(
                item['question'],
                item['embedding']
            )
            stored_questions.append({
                'id': question_id,
                **item['question']
            })
        
        return {
            'status': 'success',
            'exam_info': result['exam_info'],
            'questions_extracted': len(stored_questions),
            'questions': stored_questions
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Cleanup
        if os.path.exists(pdf_path):
            os.remove(pdf_path)

@router.post("/search/similar")
async def search_similar_questions(
    query: str,
    limit: int = 10,
    embedding_service: GeminiEmbeddingService = Depends(get_embedding_service),
    db: AsyncSession = Depends(get_db)
):
    """Search for similar questions using semantic search"""
    
    # Generate embedding for query
    query_embedding = await embedding_service.generate_embedding(
        query,
        task_type="RETRIEVAL_QUERY"
    )
    
    # Search in database
    db_ops = VectorDatabaseOperations(db)
    similar_questions = await db_ops.find_similar_questions(
        query_embedding,
        threshold=0.7,
        limit=limit
    )
    
    return {
        'query': query,
        'results': similar_questions
    }
```

---

## Best Practices

### 1. Rate Limiting and Error Handling

```python
from tenacity import retry, stop_after_attempt, wait_exponential
import asyncio
from collections import deque
import time

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

class ResilientGeminiClient:
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)
        self.rate_limiter = RateLimiter(max_requests_per_minute=60)
        
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def generate_content_with_retry(self, **kwargs):
        """Generate content with retry logic"""
        await self.rate_limiter.acquire()
        
        try:
            return await self.client.models.generate_content_async(**kwargs)
        except Exception as e:
            if "429" in str(e):  # Rate limit error
                await asyncio.sleep(60)
            raise
```

### 2. Cost Monitoring

```python
class CostMonitor:
    """Monitor API usage and costs"""
    
    # Pricing as of August 2025
    PRICING = {
        'gemini-2.5-flash-lite-001': {
            'input': 0.10 / 1_000_000,   # $0.10 per 1M tokens
            'output': 0.40 / 1_000_000   # $0.40 per 1M tokens
        },
        'gemini-embedding-001': {
            'input': 0.01 / 1_000_000    # $0.01 per 1M tokens
        }
    }
    
    def __init__(self):
        self.usage = {
            'generation': {'input_tokens': 0, 'output_tokens': 0},
            'embedding': {'input_tokens': 0}
        }
        
    def track_generation(self, input_tokens: int, output_tokens: int):
        """Track generation API usage"""
        self.usage['generation']['input_tokens'] += input_tokens
        self.usage['generation']['output_tokens'] += output_tokens
        
    def track_embedding(self, input_tokens: int):
        """Track embedding API usage"""
        self.usage['embedding']['input_tokens'] += input_tokens
        
    def get_cost_summary(self) -> Dict[str, float]:
        """Calculate total costs"""
        generation_cost = (
            self.usage['generation']['input_tokens'] * 
            self.PRICING['gemini-2.5-flash-lite-001']['input'] +
            self.usage['generation']['output_tokens'] * 
            self.PRICING['gemini-2.5-flash-lite-001']['output']
        )
        
        embedding_cost = (
            self.usage['embedding']['input_tokens'] * 
            self.PRICING['gemini-embedding-001']['input']
        )
        
        return {
            'generation_cost': generation_cost,
            'embedding_cost': embedding_cost,
            'total_cost': generation_cost + embedding_cost,
            'usage_details': self.usage
        }
```

### 3. Safety Settings

```python
from google.genai import types

def get_safety_settings():
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

# Use in generation
response = await client.models.generate_content_async(
    model='gemini-2.5-flash-lite-001',
    contents=prompt,
    config=config,
    safety_settings=get_safety_settings()
)
```

---

## Summary

This implementation guide provides a complete, production-ready solution using the google-genai SDK (0.9.0) for August 2025. Key points:

1. **Use google-genai, not google-generativeai** - The old SDK is deprecated
2. **Structured output** works natively with Pydantic models
3. **768-dimensional embeddings** provide optimal balance of quality and efficiency
4. **Async operations** throughout for better performance
5. **Comprehensive error handling** and rate limiting
6. **Cost monitoring** built into the pipeline

The implementation is designed to handle real-world exam paper processing at scale while maintaining quality and cost efficiency.