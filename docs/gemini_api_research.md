# Gemini API Research and Implementation Guide

## Executive Summary

This document provides comprehensive research and analysis on effectively using Gemini APIs for the PDF Question Extractor pipeline. The research focuses on two key components:
1. **Gemini 2.5 Flash Lite** for intelligent question extraction with structured output
2. **gemini-embedding-001** for semantic search capabilities with 768-dimensional vectors

## Table of Contents
1. [Gemini 2.5 Flash Lite for Question Extraction](#gemini-25-flash-lite-for-question-extraction)
2. [Gemini Embeddings for Semantic Search](#gemini-embeddings-for-semantic-search)
3. [Implementation Strategy](#implementation-strategy)
4. [Cost Optimization](#cost-optimization)
5. [Best Practices](#best-practices)
6. [Code Examples](#code-examples)

---

## Gemini 2.5 Flash Lite for Question Extraction

### Model Capabilities
- **Context Window**: 1 million tokens (can handle entire exam papers)
- **Native Structured Output**: Direct JSON schema support
- **Cost**: $0.10/1M input tokens, $0.40/1M output tokens
- **Speed**: Fastest model in Gemini family with lowest latency
- **Features**: Native tool support, function calling, reasoning capabilities

### Key Advantages for Our Use Case
1. **Large Context Window**: Process entire PDF documents without chunking for documents <50k characters
2. **Structured Output**: Guaranteed JSON format matching our schema
3. **Cost-Effective**: Lowest pricing tier in Gemini family
4. **Low Latency**: Ideal for interactive web application

### Structured Output Implementation

```python
from google import genai
from pydantic import BaseModel
from typing import List
import enum

# Define our question schema
class QuestionType(enum.Enum):
    MCQ = "MCQ"
    ESSAY = "Essay"
    SHORT_ANSWER = "Short Answer"
    NUMERICAL = "Numerical"
    TRUE_FALSE = "True/False"

class Question(BaseModel):
    question_number: str
    marks: int
    question_text: str
    topics: List[str]
    question_type: QuestionType

class ExamPaper(BaseModel):
    year: str
    level: str
    source_pdf: str
    questions: List[Question]

# Initialize client
client = genai.Client(api_key=GOOGLE_API_KEY)

def extract_questions(markdown_text: str, pdf_filename: str) -> ExamPaper:
    """Extract questions from OCR markdown using structured output"""
    
    prompt = f"""
    Extract all questions from the following exam paper content.
    
    Instructions:
    1. Identify each question with its number and marks
    2. Extract the complete question text
    3. Determine the question type (MCQ, Essay, Short Answer, etc.)
    4. Identify relevant topics/subtopics
    5. Extract the year and level from the content if available
    
    Content:
    {markdown_text}
    """
    
    response = client.models.generate_content(
        model='gemini-2.5-flash-lite-001',
        contents=prompt,
        config={
            'response_mime_type': 'application/json',
            'response_schema': ExamPaper,
            'temperature': 0.1,  # Lower temperature for consistency
            'top_p': 0.95,
            'max_output_tokens': 8192  # Sufficient for most exam papers
        }
    )
    
    # Parse and validate response
    return ExamPaper.model_validate_json(response.text)
```

---

## Gemini Embeddings for Semantic Search

### Model Specifications
- **Model**: gemini-embedding-001
- **Dimensions**: 768 (configurable: 768, 1536, 3072)
- **Max Tokens**: 2048 per input
- **Features**: Multilingual support, Matryoshka learning

### Why 768 Dimensions?
1. **Storage Efficiency**: 75% less storage than default 3072
2. **Performance**: Minimal quality loss with Matryoshka learning
3. **Speed**: Faster similarity searches
4. **pgvector Optimization**: Better index performance

### Implementation for Question Embeddings

```python
from google import genai
from google.genai import types
import asyncio
from typing import List, Dict

class GeminiEmbeddingService:
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)
        self.model = "gemini-embedding-001"
        self.dimension = 768
        
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        result = self.client.models.embed_content(
            model=self.model,
            contents=text,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT",
                output_dimensionality=768
            )
        )
        return result.embeddings[0].values
    
    async def generate_batch_embeddings(self, questions: List[Dict]) -> Dict[int, List[float]]:
        """Generate embeddings for multiple questions efficiently"""
        embeddings = {}
        
        # Process in batches to avoid rate limits
        batch_size = 10
        for i in range(0, len(questions), batch_size):
            batch = questions[i:i + batch_size]
            
            # Create enriched text for better semantic search
            tasks = []
            for question in batch:
                enriched_text = self._create_enriched_text(question)
                tasks.append(self.generate_embedding(enriched_text))
            
            # Process batch concurrently
            batch_embeddings = await asyncio.gather(*tasks)
            
            for j, embedding in enumerate(batch_embeddings):
                embeddings[batch[j]['id']] = embedding
                
            # Small delay to respect rate limits
            await asyncio.sleep(0.1)
            
        return embeddings
    
    def _create_enriched_text(self, question: Dict) -> str:
        """Create enriched text representation for better embeddings"""
        return f"""
        Question: {question['question_text']}
        Type: {question['question_type']}
        Topics: {', '.join(question['topics'])}
        Level: {question['level']}
        Year: {question['year']}
        Marks: {question['marks']}
        """
```

---

## Implementation Strategy

### 1. Smart Document Processing Pipeline

```python
class SmartDocumentProcessor:
    def __init__(self, llm_service, embedding_service):
        self.llm = llm_service
        self.embedding = embedding_service
        self.chunk_size = 50000  # Characters for direct processing
        
    async def process_document(self, markdown_text: str, filename: str):
        """Process document with smart chunking strategy"""
        
        # Step 1: Determine processing strategy
        if len(markdown_text) < self.chunk_size:
            # Process entire document in one pass
            exam_paper = await self._process_single_chunk(markdown_text, filename)
        else:
            # Use recursive splitting with context overlap
            exam_paper = await self._process_with_chunks(markdown_text, filename)
        
        # Step 2: Generate embeddings for all questions
        embeddings = await self.embedding.generate_batch_embeddings(
            exam_paper.questions
        )
        
        return exam_paper, embeddings
    
    async def _process_single_chunk(self, text: str, filename: str):
        """Process document in single API call"""
        return self.llm.extract_questions(text, filename)
    
    async def _process_with_chunks(self, text: str, filename: str):
        """Process large documents with smart chunking"""
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=40000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""],
            keep_separator=True
        )
        
        chunks = splitter.split_text(text)
        all_questions = []
        
        for i, chunk in enumerate(chunks):
            # Add context hints for better extraction
            contextualized_chunk = f"""
            [Document Part {i+1} of {len(chunks)}]
            {chunk}
            """
            
            partial_result = await self._process_single_chunk(
                contextualized_chunk, filename
            )
            all_questions.extend(partial_result.questions)
        
        # Deduplicate and merge results
        return self._merge_results(all_questions, filename)
```

### 2. Retry and Error Handling

```python
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import httpx

class ResilientGeminiClient:
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)
        
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.HTTPStatusError))
    )
    async def generate_content_with_retry(self, **kwargs):
        """Generate content with automatic retry logic"""
        try:
            response = await self.client.models.generate_content(**kwargs)
            return response
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:  # Rate limit
                await asyncio.sleep(60)  # Wait longer for rate limits
            raise
```

---

## Cost Optimization

### 1. Token Usage Estimation

For a typical exam paper:
- **Input tokens**: ~10,000 tokens (5-10 pages of text)
- **Output tokens**: ~2,000 tokens (structured questions)
- **Cost per paper**: ~$0.002 (input) + ~$0.0008 (output) = **$0.0028**

For 1,000 exam papers:
- **Total cost**: ~$2.80 for extraction
- **Embeddings**: Minimal additional cost

### 2. Optimization Strategies

```python
class CostOptimizedProcessor:
    def __init__(self):
        self.cache = {}  # Simple in-memory cache
        
    async def process_with_caching(self, content_hash: str, process_func):
        """Cache results to avoid reprocessing"""
        if content_hash in self.cache:
            return self.cache[content_hash]
        
        result = await process_func()
        self.cache[content_hash] = result
        return result
    
    def optimize_prompt(self, text: str) -> str:
        """Minimize prompt tokens while maintaining quality"""
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Truncate very long repeated sections
        # (implement smart truncation logic)
        
        return text
    
    async def batch_process_similar_documents(self, documents: List[str]):
        """Process similar documents together for efficiency"""
        # Group by document type/format
        grouped = self._group_similar_documents(documents)
        
        results = []
        for group in grouped:
            # Use same prompt template for similar documents
            template = self._get_optimized_template(group[0])
            
            for doc in group:
                result = await self.process_with_template(doc, template)
                results.append(result)
                
        return results
```

---

## Best Practices

### 1. Prompt Engineering for Structured Output

```python
EXTRACTION_PROMPT_TEMPLATE = """
You are an expert at extracting questions from exam papers.

CRITICAL INSTRUCTIONS:
1. Extract EVERY question, including sub-questions (e.g., 1a, 1b, 1c)
2. Preserve the EXACT question text, including mathematical notation
3. Identify question type based on answer format expected
4. Extract marks from patterns like "[5 marks]", "(5)", "5M"
5. Infer topics from question content and context

QUESTION TYPE CLASSIFICATION:
- MCQ: Multiple choice with options (A, B, C, D)
- Essay: Requires extended written response (usually >5 marks)
- Short Answer: Brief response expected (1-5 marks)
- Numerical: Calculation or numerical answer required
- True/False: Binary choice questions

Document content:
{content}

Extract all questions following the exact schema provided.
"""
```

### 2. Embedding Quality Enhancement

```python
def enhance_embedding_quality(question: Dict) -> str:
    """Create semantically rich text for embeddings"""
    
    # Add semantic markers
    type_descriptors = {
        "MCQ": "multiple choice selection",
        "Essay": "extended written response analysis",
        "Short Answer": "brief explanation",
        "Numerical": "mathematical calculation",
        "True/False": "binary classification"
    }
    
    # Build enhanced representation
    enhanced = f"""
    Educational Question:
    Content: {question['question_text']}
    Answer Format: {type_descriptors.get(question['question_type'], question['question_type'])}
    Subject Areas: {', '.join(question['topics'])}
    Difficulty Level: {question['level']}
    Assessment Year: {question['year']}
    Point Value: {question['marks']} marks
    Knowledge Type: {_infer_knowledge_type(question)}
    """
    
    return enhanced

def _infer_knowledge_type(question: Dict) -> str:
    """Infer cognitive level from question text"""
    text_lower = question['question_text'].lower()
    
    if any(word in text_lower for word in ['analyze', 'evaluate', 'compare']):
        return "analytical thinking"
    elif any(word in text_lower for word in ['explain', 'describe', 'discuss']):
        return "conceptual understanding"
    elif any(word in text_lower for word in ['calculate', 'solve', 'derive']):
        return "computational skills"
    elif any(word in text_lower for word in ['define', 'list', 'state']):
        return "factual recall"
    else:
        return "general knowledge"
```

### 3. Rate Limiting and Quota Management

```python
import time
from collections import deque

class RateLimiter:
    def __init__(self, max_requests_per_minute=60):
        self.max_requests = max_requests_per_minute
        self.requests = deque()
        
    async def acquire(self):
        """Wait if necessary to respect rate limits"""
        now = time.time()
        
        # Remove requests older than 1 minute
        while self.requests and self.requests[0] < now - 60:
            self.requests.popleft()
        
        # If at limit, wait
        if len(self.requests) >= self.max_requests:
            sleep_time = 60 - (now - self.requests[0]) + 0.1
            await asyncio.sleep(sleep_time)
            await self.acquire()  # Recursive call
        else:
            self.requests.append(now)
```

---

## Integration with Existing Pipeline

### 1. Service Integration

```python
# services/llm_service.py
class GeminiLLMService:
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)
        self.rate_limiter = RateLimiter()
        
    async def extract_questions(self, markdown_text: str, pdf_filename: str):
        await self.rate_limiter.acquire()
        
        # Use optimized processing
        processor = SmartDocumentProcessor(self, None)
        return await processor.process_document(markdown_text, pdf_filename)

# services/embedding_service.py  
class GeminiEmbeddingService:
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)
        self.rate_limiter = RateLimiter(max_requests_per_minute=100)
        
    async def generate_embeddings_for_questions(self, questions: List[Dict]):
        embeddings = {}
        
        for question in questions:
            await self.rate_limiter.acquire()
            
            enhanced_text = enhance_embedding_quality(question)
            embedding = await self.generate_embedding(enhanced_text)
            embeddings[question['id']] = embedding
            
        return embeddings
```

### 2. Database Storage with pgvector

```python
from sqlalchemy import text
import numpy as np

async def store_question_with_embedding(db_session, question: Dict, embedding: List[float]):
    """Store question and its embedding atomically"""
    
    async with db_session.begin():
        # Insert question
        result = await db_session.execute(
            text("""
                INSERT INTO questions (
                    question_number, marks, year, level, topics,
                    question_type, question_text, source_pdf
                ) VALUES (
                    :q_num, :marks, :year, :level, :topics,
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
        await db_session.execute(
            text("""
                INSERT INTO question_embeddings (
                    question_id, embedding, model_name, model_version
                ) VALUES (
                    :q_id, :embedding::vector, :model, :version
                )
            """),
            {
                'q_id': question_id,
                'embedding': embedding,
                'model': 'gemini-embedding-001',
                'version': '1.0'
            }
        )
```

---

## Performance Metrics and Monitoring

### Key Metrics to Track
1. **Token Usage**: Input/output tokens per document
2. **Processing Time**: Time per document and per question
3. **Error Rates**: Failed extractions, retries needed
4. **Quality Metrics**: Questions extracted vs expected
5. **Cost per Document**: Actual vs estimated

### Monitoring Implementation

```python
import logging
from datetime import datetime

class PerformanceMonitor:
    def __init__(self):
        self.metrics = {
            'documents_processed': 0,
            'questions_extracted': 0,
            'total_input_tokens': 0,
            'total_output_tokens': 0,
            'errors': 0,
            'processing_time': []
        }
        
    async def track_processing(self, func, *args, **kwargs):
        """Track performance metrics for processing"""
        start_time = datetime.now()
        
        try:
            result = await func(*args, **kwargs)
            
            # Update metrics
            self.metrics['documents_processed'] += 1
            if hasattr(result, 'questions'):
                self.metrics['questions_extracted'] += len(result.questions)
                
            processing_time = (datetime.now() - start_time).total_seconds()
            self.metrics['processing_time'].append(processing_time)
            
            return result
            
        except Exception as e:
            self.metrics['errors'] += 1
            logging.error(f"Processing error: {e}")
            raise
            
    def get_summary(self):
        """Get performance summary"""
        avg_time = sum(self.metrics['processing_time']) / len(self.metrics['processing_time']) if self.metrics['processing_time'] else 0
        
        return {
            'documents_processed': self.metrics['documents_processed'],
            'questions_extracted': self.metrics['questions_extracted'],
            'average_processing_time': avg_time,
            'error_rate': self.metrics['errors'] / max(self.metrics['documents_processed'], 1),
            'estimated_cost': self._calculate_cost()
        }
    
    def _calculate_cost(self):
        """Calculate estimated cost based on token usage"""
        input_cost = (self.metrics['total_input_tokens'] / 1_000_000) * 0.10
        output_cost = (self.metrics['total_output_tokens'] / 1_000_000) * 0.40
        return input_cost + output_cost
```

---

## Conclusion

The Gemini API suite provides an excellent foundation for our PDF Question Extractor pipeline:

1. **Gemini 2.5 Flash Lite** offers the perfect balance of cost, speed, and capability for question extraction with native structured output support

2. **gemini-embedding-001** with 768 dimensions provides efficient semantic search while maintaining high quality through Matryoshka learning

3. **Cost-effective**: Total processing cost of ~$2.80 per 1000 exam papers

4. **Scalable**: Built-in rate limiting, retry logic, and batch processing support

5. **Quality-focused**: Enhanced embeddings and smart chunking ensure high-quality results

The implementation strategies outlined provide a robust, production-ready solution that can handle the complexities of real-world exam paper processing while maintaining cost efficiency and performance.