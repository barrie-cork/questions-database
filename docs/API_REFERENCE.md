# PDF Question Extractor - API Reference

Complete API reference with examples, schemas, and integration guides.

## 📋 Table of Contents

1. [Overview](#overview)
2. [Authentication](#authentication)
3. [Base Configuration](#base-configuration)
4. [Endpoints](#endpoints)
   - [Health & Status](#health--status)
   - [PDF Processing](#pdf-processing)
   - [Question Management](#question-management)
   - [Search & Similarity](#search--similarity)
   - [Export & Analytics](#export--analytics)
5. [WebSocket API](#websocket-api)
6. [Response Schemas](#response-schemas)
7. [Error Handling](#error-handling)
8. [Rate Limiting](#rate-limiting)
9. [Code Examples](#code-examples)
10. [Best Practices](#best-practices)

## Overview

The PDF Question Extractor API is a RESTful API built with FastAPI that provides comprehensive functionality for extracting, managing, and searching exam questions from PDF files.

### Key Features
- 🚀 Async/await architecture for high performance
- 📖 Auto-generated OpenAPI documentation
- 🔄 Real-time updates via WebSocket
- 🎯 Type-safe with Pydantic models
- 🔍 Semantic search with vector embeddings

### API Versioning
- Current Version: `1.0.0`
- Base URL: `http://localhost:8000/api`
- Documentation: `http://localhost:8000/api/docs`

## Authentication

Currently, the API uses environment-based API keys for external services. Future versions will implement JWT authentication.

```python
# Current: Set in environment
MISTRAL_API_KEY=your_mistral_key
GOOGLE_API_KEY=your_google_key

# Future: JWT Bearer token
Authorization: Bearer <token>
```

## Base Configuration

### Request Headers
```http
Content-Type: application/json
Accept: application/json
```

### Response Headers
```http
Content-Type: application/json
X-Request-ID: <unique-request-id>
X-Processing-Time: <milliseconds>
```

## Endpoints

### Health & Status

#### GET /health
Check system health and service availability.

**Response Example:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-20T10:30:00Z",
  "version": "1.0.0",
  "database_connected": true,
  "services": {
    "database": "healthy",
    "ocr_service": "configured",
    "llm_service": "configured",
    "embedding_service": "configured"
  }
}
```

**Status Codes:**
- `200` - System healthy
- `503` - System degraded or unavailable

### PDF Processing

#### POST /api/upload
Upload one or more PDF files for processing.

**Request:**
```bash
curl -X POST "http://localhost:8000/api/upload" \
  -H "accept: application/json" \
  -F "pdfs=@exam1.pdf" \
  -F "pdfs=@exam2.pdf"
```

**Response:**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "pending",
  "total_files": 2,
  "message": "Processing started",
  "websocket_url": "ws://localhost:8000/ws/processing/550e8400-e29b-41d4-a716-446655440000"
}
```

**Constraints:**
- Max file size: 50MB per file
- Max files per request: 10
- Supported formats: PDF only

#### GET /api/process/{job_id}
Get the status of a processing job.

**Request:**
```bash
curl "http://localhost:8000/api/process/550e8400-e29b-41d4-a716-446655440000"
```

**Response:**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "processing",
  "total_files": 2,
  "processed_files": 1,
  "total_questions": 25,
  "files": [
    {
      "filename": "exam1.pdf",
      "status": "completed",
      "questions_extracted": 25,
      "processing_time_ms": 3500,
      "error": null
    },
    {
      "filename": "exam2.pdf",
      "status": "processing",
      "questions_extracted": 0,
      "processing_time_ms": null,
      "error": null
    }
  ],
  "started_at": "2024-01-20T10:30:00Z",
  "completed_at": null,
  "estimated_completion": "2024-01-20T10:32:00Z"
}
```

### Question Management

#### GET /api/questions
Retrieve paginated list of extracted questions.

**Query Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| page | integer | 1 | Page number |
| per_page | integer | 20 | Items per page (max: 100) |
| status | string | all | Filter: pending, approved, rejected |
| source_pdf | string | - | Filter by source PDF filename |
| year | string | - | Filter by year |
| level | string | - | Filter by education level |
| question_type | string | - | Filter by question type |
| topics | array | - | Filter by topics (comma-separated) |
| search | string | - | Full-text search query |
| sort_by | string | id | Sort field |
| sort_order | string | asc | Sort order: asc, desc |

**Request:**
```bash
curl "http://localhost:8000/api/questions?page=1&per_page=20&status=pending&year=2023"
```

**Response:**
```json
{
  "total": 150,
  "page": 1,
  "per_page": 20,
  "total_pages": 8,
  "questions": [
    {
      "id": 1,
      "question_number": "1a",
      "marks": 5,
      "year": "2023",
      "level": "A-Level",
      "topics": ["Calculus", "Integration"],
      "question_type": "Short Answer",
      "question_text": "Find the integral of x^2 + 2x + 1",
      "source_pdf": "math_2023.pdf",
      "status": "pending",
      "modified": false,
      "extraction_date": "2024-01-20T10:00:00Z",
      "metadata": {
        "page_number": 2,
        "confidence_score": 0.95
      }
    }
  ]
}
```

#### GET /api/questions/{id}
Get details of a specific question.

**Request:**
```bash
curl "http://localhost:8000/api/questions/1"
```

**Response:**
```json
{
  "id": 1,
  "question_number": "1a",
  "marks": 5,
  "year": "2023",
  "level": "A-Level",
  "topics": ["Calculus", "Integration"],
  "question_type": "Short Answer",
  "question_text": "Find the integral of x^2 + 2x + 1",
  "source_pdf": "math_2023.pdf",
  "status": "pending",
  "modified": false,
  "extraction_date": "2024-01-20T10:00:00Z",
  "metadata": {
    "page_number": 2,
    "confidence_score": 0.95,
    "extraction_method": "gemini-2.5-flash",
    "ocr_quality": "high"
  }
}
```

#### PUT /api/questions/{id}
Update a question's details.

**Request:**
```bash
curl -X PUT "http://localhost:8000/api/questions/1" \
  -H "Content-Type: application/json" \
  -d '{
    "question_text": "Find the integral of x² + 2x + 1 with respect to x",
    "marks": 6,
    "topics": ["Calculus", "Integration", "Polynomials"],
    "status": "approved"
  }'
```

**Response:**
```json
{
  "id": 1,
  "success": true,
  "message": "Question updated successfully",
  "modified": true
}
```

#### POST /api/questions/bulk
Perform bulk operations on multiple questions.

**Operations:**
- `update_status` - Change status of multiple questions
- `delete` - Delete multiple questions
- `update_topics` - Update topics for multiple questions
- `approve_all` - Approve multiple questions
- `reject_all` - Reject multiple questions

**Request:**
```bash
curl -X POST "http://localhost:8000/api/questions/bulk" \
  -H "Content-Type: application/json" \
  -d '{
    "operation": "update_status",
    "question_ids": [1, 2, 3, 4, 5],
    "data": {
      "status": "approved"
    }
  }'
```

**Response:**
```json
{
  "success": true,
  "operation": "update_status",
  "total": 5,
  "succeeded": 5,
  "failed": 0,
  "errors": [],
  "message": "Bulk operation completed successfully"
}
```

#### POST /api/questions/save
Save approved questions to permanent storage with embeddings.

**Request:**
```bash
curl -X POST "http://localhost:8000/api/questions/save" \
  -H "Content-Type: application/json" \
  -d '{
    "question_ids": [1, 3, 5, 7, 9]
  }'
```

**Response:**
```json
{
  "success": true,
  "total": 5,
  "saved": 5,
  "embeddings_generated": 5,
  "failed": 0,
  "errors": [],
  "message": "Questions saved to permanent storage"
}
```

### Search & Similarity

#### POST /api/questions/search
Semantic search using natural language queries.

**Request:**
```bash
curl -X POST "http://localhost:8000/api/questions/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "integration by parts",
    "limit": 10,
    "similarity_threshold": 0.7,
    "filters": {
      "year": "2023",
      "level": "A-Level",
      "topics": ["Calculus"]
    }
  }'
```

**Response:**
```json
{
  "query": "integration by parts",
  "total_results": 5,
  "results": [
    {
      "id": 42,
      "question_text": "Use integration by parts to evaluate ∫x·sin(x)dx",
      "similarity_score": 0.92,
      "topics": ["Calculus", "Integration"],
      "year": "2023",
      "marks": 8,
      "source_pdf": "calculus_2023.pdf"
    }
  ],
  "search_time_ms": 45
}
```

#### POST /api/questions/similar/{id}
Find questions similar to a given question.

**Request:**
```bash
curl -X POST "http://localhost:8000/api/questions/similar/42" \
  -H "Content-Type: application/json" \
  -d '{
    "limit": 5,
    "similarity_threshold": 0.8
  }'
```

**Response:**
```json
{
  "source_question_id": 42,
  "source_question_text": "Use integration by parts to evaluate ∫x·sin(x)dx",
  "similar_questions": [
    {
      "id": 156,
      "question_text": "Apply integration by parts to find ∫x·cos(x)dx",
      "similarity_score": 0.89,
      "source_pdf": "practice_2022.pdf",
      "year": "2022"
    }
  ]
}
```

### Export & Analytics

#### GET /api/export
Export questions in various formats.

**Query Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| format | string | csv | Export format: csv, json, xlsx |
| status | string | approved | Filter by status |
| ids | string | - | Comma-separated question IDs |
| include_metadata | boolean | false | Include metadata in export |

**Request:**
```bash
curl "http://localhost:8000/api/export?format=csv&status=approved" \
  -o approved_questions.csv
```

**Response:**
- For CSV/XLSX: Binary file download
- For JSON:
```json
{
  "export_date": "2024-01-20T10:30:00Z",
  "export_format": "json",
  "total_questions": 50,
  "filters_applied": {
    "status": "approved"
  },
  "questions": [
    {
      "id": 1,
      "question_number": "1a",
      "question_text": "...",
      "marks": 5,
      "year": "2023",
      "level": "A-Level",
      "topics": ["Calculus"],
      "question_type": "Short Answer",
      "source_pdf": "math_2023.pdf"
    }
  ]
}
```

#### GET /api/stats
Get question statistics and analytics.

**Request:**
```bash
curl "http://localhost:8000/api/stats"
```

**Response:**
```json
{
  "generated_at": "2024-01-20T10:30:00Z",
  "total_questions": 500,
  "total_approved": 400,
  "by_status": {
    "pending": 50,
    "approved": 400,
    "rejected": 50
  },
  "by_year": {
    "2023": 250,
    "2022": 150,
    "2021": 100
  },
  "by_type": {
    "MCQ": 200,
    "Essay": 100,
    "Short Answer": 150,
    "Numerical": 50
  },
  "by_level": {
    "A-Level": 300,
    "GCSE": 200
  },
  "top_topics": [
    {"topic": "Calculus", "count": 120},
    {"topic": "Algebra", "count": 80},
    {"topic": "Geometry", "count": 60}
  ],
  "processing_stats": {
    "total_pdfs_processed": 50,
    "average_questions_per_pdf": 10,
    "success_rate": 0.98
  }
}
```

## WebSocket API

### Connection
Connect to receive real-time processing updates.

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/processing/550e8400-e29b-41d4-a716-446655440000');

ws.onopen = () => {
    console.log('Connected to processing updates');
};

ws.onmessage = (event) => {
    const update = JSON.parse(event.data);
    console.log('Progress update:', update);
};

ws.onerror = (error) => {
    console.error('WebSocket error:', error);
};

ws.onclose = () => {
    console.log('Processing complete or connection closed');
};
```

### Message Types

#### Progress Update
```json
{
  "type": "progress",
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "current_file": "exam_2023.pdf",
  "current_step": "extracting_questions",
  "processed_files": 1,
  "total_files": 3,
  "percentage": 33.3,
  "message": "Extracting questions from exam_2023.pdf"
}
```

#### File Completed
```json
{
  "type": "file_completed",
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "filename": "exam_2023.pdf",
  "status": "success",
  "questions_extracted": 25,
  "processing_time_ms": 3500,
  "steps_completed": [
    {"step": "ocr", "duration_ms": 2000},
    {"step": "extraction", "duration_ms": 1000},
    {"step": "storage", "duration_ms": 500}
  ]
}
```

#### Job Completed
```json
{
  "type": "job_completed",
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "success",
  "total_files": 3,
  "successful_files": 3,
  "failed_files": 0,
  "total_questions": 75,
  "total_duration_ms": 10500
}
```

#### Error
```json
{
  "type": "error",
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "filename": "corrupted.pdf",
  "error_code": "OCR_FAILED",
  "error_message": "Failed to extract text from PDF: Invalid PDF format",
  "recoverable": false
}
```

## Response Schemas

### Question Schema
```python
class Question(BaseModel):
    id: int
    question_number: str
    marks: int
    year: str
    level: str
    topics: List[str]
    question_type: str
    question_text: str
    source_pdf: str
    status: str
    modified: bool
    extraction_date: datetime
    metadata: Optional[Dict[str, Any]]
```

### Pagination Schema
```python
class PaginatedResponse(BaseModel):
    total: int
    page: int
    per_page: int
    total_pages: int
    questions: List[Question]
```

### Error Schema
```python
class ErrorResponse(BaseModel):
    error: Dict[str, Any]
    request_id: str
    timestamp: datetime
```

## Error Handling

### Error Response Format
```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid request data",
    "details": {
      "field": "marks",
      "error": "Must be a positive integer"
    }
  },
  "request_id": "req_550e8400",
  "timestamp": "2024-01-20T10:30:00Z"
}
```

### Error Codes
| Code | Description | HTTP Status |
|------|-------------|-------------|
| VALIDATION_ERROR | Invalid request data | 422 |
| NOT_FOUND | Resource not found | 404 |
| PROCESSING_ERROR | PDF processing failed | 500 |
| OCR_ERROR | OCR extraction failed | 500 |
| LLM_ERROR | Question extraction failed | 500 |
| DATABASE_ERROR | Database operation failed | 500 |
| RATE_LIMIT | Too many requests | 429 |
| FILE_TOO_LARGE | File exceeds size limit | 413 |
| UNSUPPORTED_FORMAT | Not a valid PDF | 415 |

### Handling Errors
```python
import requests

try:
    response = requests.post("http://localhost:8000/api/upload", files=files)
    response.raise_for_status()
    data = response.json()
except requests.exceptions.HTTPError as e:
    if e.response.status_code == 422:
        errors = e.response.json()
        print(f"Validation error: {errors['error']['message']}")
    elif e.response.status_code == 429:
        print("Rate limited. Try again later.")
    else:
        print(f"HTTP error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Rate Limiting

### Limits
| Endpoint | Limit | Window |
|----------|-------|--------|
| /api/upload | 10 requests | 1 minute |
| /api/questions/* | 100 requests | 1 minute |
| /api/export | 5 requests | 1 minute |
| /api/search | 30 requests | 1 minute |

### Rate Limit Headers
```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1705750200
X-RateLimit-Window: 60
```

### Handling Rate Limits
```python
def handle_rate_limit(response):
    if response.status_code == 429:
        reset_time = int(response.headers.get('X-RateLimit-Reset', 0))
        wait_time = reset_time - int(time.time())
        print(f"Rate limited. Wait {wait_time} seconds.")
        time.sleep(wait_time)
        return True
    return False
```

## Code Examples

### Python Client Example
```python
import asyncio
import aiohttp
import aiofiles

class PDFQuestionExtractorClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.session.close()
    
    async def upload_pdf(self, file_path):
        """Upload a PDF for processing"""
        async with aiofiles.open(file_path, 'rb') as f:
            data = aiohttp.FormData()
            data.add_field('pdfs', f, filename=file_path.name)
            
            async with self.session.post(
                f"{self.base_url}/api/upload",
                data=data
            ) as response:
                return await response.json()
    
    async def get_questions(self, page=1, per_page=20, **filters):
        """Get paginated questions"""
        params = {"page": page, "per_page": per_page, **filters}
        async with self.session.get(
            f"{self.base_url}/api/questions",
            params=params
        ) as response:
            return await response.json()
    
    async def search_questions(self, query, limit=10):
        """Search questions semantically"""
        async with self.session.post(
            f"{self.base_url}/api/questions/search",
            json={"query": query, "limit": limit}
        ) as response:
            return await response.json()

# Usage
async def main():
    async with PDFQuestionExtractorClient() as client:
        # Upload PDF
        result = await client.upload_pdf("exam.pdf")
        print(f"Job ID: {result['job_id']}")
        
        # Get questions
        questions = await client.get_questions(status="approved")
        print(f"Found {questions['total']} questions")
        
        # Search
        results = await client.search_questions("calculus integration")
        for r in results['results']:
            print(f"- {r['question_text']} (score: {r['similarity_score']})")

asyncio.run(main())
```

### JavaScript Client Example
```javascript
class PDFQuestionExtractorClient {
    constructor(baseUrl = 'http://localhost:8000') {
        this.baseUrl = baseUrl;
    }
    
    async uploadPDF(file) {
        const formData = new FormData();
        formData.append('pdfs', file);
        
        const response = await fetch(`${this.baseUrl}/api/upload`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`Upload failed: ${response.statusText}`);
        }
        
        return response.json();
    }
    
    async getQuestions(params = {}) {
        const queryString = new URLSearchParams(params).toString();
        const response = await fetch(
            `${this.baseUrl}/api/questions?${queryString}`
        );
        
        return response.json();
    }
    
    async updateQuestion(id, data) {
        const response = await fetch(
            `${this.baseUrl}/api/questions/${id}`,
            {
                method: 'PUT',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(data)
            }
        );
        
        return response.json();
    }
    
    connectWebSocket(jobId, callbacks) {
        const ws = new WebSocket(
            `ws://localhost:8000/ws/processing/${jobId}`
        );
        
        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            
            switch(data.type) {
                case 'progress':
                    callbacks.onProgress?.(data);
                    break;
                case 'file_completed':
                    callbacks.onFileCompleted?.(data);
                    break;
                case 'job_completed':
                    callbacks.onJobCompleted?.(data);
                    break;
                case 'error':
                    callbacks.onError?.(data);
                    break;
            }
        };
        
        ws.onerror = callbacks.onError;
        ws.onclose = callbacks.onClose;
        
        return ws;
    }
}

// Usage
const client = new PDFQuestionExtractorClient();

// Upload and monitor
async function processExamPaper(file) {
    try {
        // Upload
        const uploadResult = await client.uploadPDF(file);
        console.log('Processing started:', uploadResult.job_id);
        
        // Connect WebSocket for progress
        const ws = client.connectWebSocket(uploadResult.job_id, {
            onProgress: (data) => {
                console.log(`Progress: ${data.percentage}%`);
                updateProgressBar(data.percentage);
            },
            onFileCompleted: (data) => {
                console.log(`Completed: ${data.filename}`);
                console.log(`Questions extracted: ${data.questions_extracted}`);
            },
            onJobCompleted: (data) => {
                console.log('All files processed!');
                console.log(`Total questions: ${data.total_questions}`);
                loadQuestions();
            },
            onError: (error) => {
                console.error('Processing error:', error);
                showError(error.error_message);
            }
        });
        
    } catch (error) {
        console.error('Upload failed:', error);
    }
}

// Get and display questions
async function loadQuestions() {
    const questions = await client.getQuestions({
        status: 'pending',
        page: 1,
        per_page: 50
    });
    
    displayQuestionsInTable(questions.questions);
}
```

### CURL Examples Collection
```bash
# Health check
curl http://localhost:8000/health

# Upload multiple PDFs
curl -X POST http://localhost:8000/api/upload \
  -F "pdfs=@math_2023.pdf" \
  -F "pdfs=@physics_2023.pdf"

# Get pending questions from specific PDF
curl "http://localhost:8000/api/questions?status=pending&source_pdf=math_2023.pdf"

# Update question with all fields
curl -X PUT http://localhost:8000/api/questions/1 \
  -H "Content-Type: application/json" \
  -d '{
    "question_number": "1a",
    "marks": 10,
    "year": "2023",
    "level": "A-Level",
    "topics": ["Calculus", "Differentiation", "Chain Rule"],
    "question_type": "Long Answer",
    "question_text": "Differentiate y = sin(x²) with respect to x, showing all steps.",
    "status": "approved"
  }'

# Bulk approve questions
curl -X POST http://localhost:8000/api/questions/bulk \
  -H "Content-Type: application/json" \
  -d '{
    "operation": "approve_all",
    "question_ids": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
  }'

# Search for calculus questions
curl -X POST http://localhost:8000/api/questions/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "differentiation chain rule",
    "limit": 20,
    "filters": {
      "level": "A-Level",
      "year": "2023"
    }
  }'

# Export approved questions as CSV
curl "http://localhost:8000/api/export?format=csv&status=approved" \
  -o approved_questions_2023.csv

# Get statistics
curl http://localhost:8000/api/stats
```

## Best Practices

### 1. Error Handling
Always implement proper error handling:
```python
try:
    response = await client.post("/api/upload", files=files)
    response.raise_for_status()
except httpx.HTTPStatusError as e:
    if e.response.status_code == 422:
        # Handle validation errors
        errors = e.response.json()
    elif e.response.status_code == 429:
        # Handle rate limiting
        retry_after = e.response.headers.get('X-RateLimit-Reset')
    else:
        # Handle other HTTP errors
        pass
except Exception as e:
    # Handle network errors
    pass
```

### 2. Pagination
Always use pagination for large datasets:
```python
async def get_all_questions(client):
    all_questions = []
    page = 1
    
    while True:
        response = await client.get_questions(page=page, per_page=100)
        all_questions.extend(response['questions'])
        
        if page >= response['total_pages']:
            break
        page += 1
    
    return all_questions
```

### 3. WebSocket Reconnection
Implement reconnection logic:
```javascript
class ReconnectingWebSocket {
    constructor(url, options = {}) {
        this.url = url;
        this.reconnectInterval = options.reconnectInterval || 5000;
        this.shouldReconnect = true;
        this.connect();
    }
    
    connect() {
        this.ws = new WebSocket(this.url);
        
        this.ws.onopen = () => {
            console.log('WebSocket connected');
            this.reconnectInterval = 5000; // Reset interval
        };
        
        this.ws.onclose = () => {
            if (this.shouldReconnect) {
                console.log(`Reconnecting in ${this.reconnectInterval}ms...`);
                setTimeout(() => this.connect(), this.reconnectInterval);
                this.reconnectInterval = Math.min(30000, this.reconnectInterval * 2);
            }
        };
        
        // Forward other events
        this.ws.onmessage = this.onmessage;
        this.ws.onerror = this.onerror;
    }
    
    close() {
        this.shouldReconnect = false;
        this.ws.close();
    }
}
```

### 4. Batch Operations
Use batch operations when possible:
```python
# Instead of updating questions one by one
for question_id in question_ids:
    await client.update_question(question_id, {"status": "approved"})

# Use bulk operations
await client.bulk_update({
    "operation": "update_status",
    "question_ids": question_ids,
    "data": {"status": "approved"}
})
```

### 5. Caching
Implement client-side caching:
```javascript
class CachedClient extends PDFQuestionExtractorClient {
    constructor(baseUrl) {
        super(baseUrl);
        this.cache = new Map();
        this.cacheTimeout = 60000; // 1 minute
    }
    
    async getQuestions(params = {}) {
        const cacheKey = JSON.stringify(params);
        const cached = this.cache.get(cacheKey);
        
        if (cached && Date.now() - cached.timestamp < this.cacheTimeout) {
            return cached.data;
        }
        
        const data = await super.getQuestions(params);
        this.cache.set(cacheKey, {
            data,
            timestamp: Date.now()
        });
        
        return data;
    }
}
```

### 6. Request Timeouts
Set appropriate timeouts:
```python
async with aiohttp.ClientSession(
    timeout=aiohttp.ClientTimeout(total=30)
) as session:
    # Use session for requests
    pass
```

### 7. Monitoring
Log important events:
```python
import logging

logger = logging.getLogger(__name__)

async def process_with_monitoring(client, file_path):
    start_time = time.time()
    
    try:
        result = await client.upload_pdf(file_path)
        duration = time.time() - start_time
        
        logger.info(f"PDF processed successfully", extra={
            "file": file_path,
            "job_id": result['job_id'],
            "duration": duration
        })
        
        return result
    except Exception as e:
        logger.error(f"PDF processing failed", extra={
            "file": file_path,
            "error": str(e),
            "duration": time.time() - start_time
        })
        raise
```

## API Testing

### Using the Interactive Docs
1. Navigate to http://localhost:8000/api/docs
2. Click on any endpoint to expand
3. Click "Try it out"
4. Fill in parameters
5. Click "Execute"
6. View the response

### Postman Collection
Import this collection for easy testing:
```json
{
  "info": {
    "name": "PDF Question Extractor API",
    "version": "1.0.0"
  },
  "item": [
    {
      "name": "Health Check",
      "request": {
        "method": "GET",
        "url": "{{base_url}}/health"
      }
    },
    {
      "name": "Upload PDF",
      "request": {
        "method": "POST",
        "url": "{{base_url}}/api/upload",
        "body": {
          "mode": "formdata",
          "formdata": [
            {
              "key": "pdfs",
              "type": "file",
              "src": "exam.pdf"
            }
          ]
        }
      }
    }
  ],
  "variable": [
    {
      "key": "base_url",
      "value": "http://localhost:8000"
    }
  ]
}
```

## Migration Guide

### From v0.x to v1.0
1. Update endpoint URLs (add `/api` prefix)
2. Update response parsing (new schema structure)
3. Implement WebSocket for progress (replacing polling)
4. Update error handling (new error format)

---

*For more examples and use cases, check the [test files](../pdf_question_extractor/tests/) in the repository.*