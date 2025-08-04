# API Design - PDF Question Extractor

## Overview
RESTful API built with FastAPI providing endpoints for PDF processing, question management, and real-time updates via WebSocket.

## Base Configuration

### Base URL
```
http://localhost:8000/api
```

### Authentication
MVP uses API key authentication via environment variables. Future versions will implement JWT.

### Common Headers
```http
Content-Type: application/json
Accept: application/json
```

## API Endpoints

### 1. Health Check

#### GET /health
Check application and database health.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T12:00:00Z",
  "services": {
    "database": "connected",
    "mistral_api": "available",
    "gemini_api": "available"
  }
}
```

### 2. PDF Upload & Processing

#### POST /api/upload
Upload PDF files for processing.

**Request:**
```http
Content-Type: multipart/form-data

pdfs: [file1.pdf, file2.pdf, ...]
```

**Response:**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "pending",
  "total_files": 5,
  "message": "Processing started",
  "websocket_url": "ws://localhost:8000/ws/processing/550e8400-e29b-41d4-a716-446655440000"
}
```

#### GET /api/process/{job_id}
Get processing job status.

**Response:**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "processing",
  "total_files": 5,
  "processed_files": 2,
  "total_questions": 45,
  "files": [
    {
      "filename": "exam_2023.pdf",
      "status": "completed",
      "questions_extracted": 25,
      "processing_time_ms": 3500
    },
    {
      "filename": "exam_2024.pdf",
      "status": "processing",
      "questions_extracted": 0
    }
  ],
  "started_at": "2024-01-01T12:00:00Z",
  "estimated_completion": "2024-01-01T12:05:00Z"
}
```

### 3. Question Management

#### GET /api/questions
Get paginated extracted questions.

**Query Parameters:**
- `page` (int): Page number (default: 1)
- `per_page` (int): Items per page (default: 20, max: 100)
- `status` (string): Filter by status (pending|approved|rejected)
- `source_pdf` (string): Filter by source PDF
- `year` (string): Filter by year
- `question_type` (string): Filter by type
- `search` (string): Full-text search

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
      "extraction_date": "2024-01-01T12:00:00Z"
    }
  ]
}
```

#### GET /api/questions/{id}
Get single question details.

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
  "extraction_date": "2024-01-01T12:00:00Z",
  "metadata": {
    "page_number": 2,
    "confidence_score": 0.95
  }
}
```

#### PUT /api/questions/{id}
Update a single question.

**Request:**
```json
{
  "question_number": "1a",
  "marks": 5,
  "year": "2023",
  "level": "A-Level", 
  "topics": ["Calculus", "Integration", "Polynomials"],
  "question_type": "Short Answer",
  "question_text": "Find the integral of x^2 + 2x + 1 with respect to x",
  "status": "approved"
}
```

**Response:**
```json
{
  "id": 1,
  "message": "Question updated successfully",
  "modified": true
}
```

#### POST /api/questions/bulk
Perform bulk operations on questions.

**Request:**
```json
{
  "operation": "update_status",
  "question_ids": [1, 2, 3, 4, 5],
  "data": {
    "status": "approved"
  }
}
```

**Response:**
```json
{
  "success": true,
  "updated": 5,
  "failed": 0,
  "message": "Bulk operation completed"
}
```

#### POST /api/questions/save
Save approved questions to permanent storage.

**Request:**
```json
{
  "question_ids": [1, 3, 5, 7, 9]
}
```

**Response:**
```json
{
  "success": true,
  "saved": 5,
  "embeddings_generated": 5,
  "message": "Questions saved to permanent storage"
}
```

### 4. Search & Similarity

#### POST /api/questions/search
Search questions with semantic similarity.

**Request:**
```json
{
  "query": "integration by parts",
  "limit": 10,
  "similarity_threshold": 0.8,
  "filters": {
    "year": "2023",
    "level": "A-Level"
  }
}
```

**Response:**
```json
{
  "total": 5,
  "results": [
    {
      "id": 42,
      "question_text": "Use integration by parts to solve...",
      "similarity_score": 0.92,
      "topics": ["Calculus", "Integration"],
      "year": "2023"
    }
  ]
}
```

#### POST /api/questions/similar/{id}
Find questions similar to a given question.

**Response:**
```json
{
  "source_question_id": 1,
  "similar_questions": [
    {
      "id": 15,
      "question_text": "Evaluate the integral of x^2 + 3x + 2",
      "similarity_score": 0.89,
      "source_pdf": "math_2022.pdf"
    }
  ]
}
```

### 5. Export

#### GET /api/export
Export questions in various formats.

**Query Parameters:**
- `format` (string): Export format (csv|json|xlsx)
- `status` (string): Filter by status
- `ids` (string): Comma-separated question IDs

**Response:**
- CSV/XLSX: Binary file download
- JSON: 
```json
{
  "export_date": "2024-01-01T12:00:00Z",
  "total_questions": 50,
  "questions": [...]
}
```

### 6. Statistics

#### GET /api/stats
Get question statistics.

**Response:**
```json
{
  "total_questions": 500,
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
    "Essay": 150,
    "Short Answer": 150
  },
  "top_topics": [
    {"topic": "Calculus", "count": 120},
    {"topic": "Algebra", "count": 80}
  ]
}
```

## WebSocket Endpoint

### WS /ws/processing/{job_id}
Real-time processing updates.

**Connection:**
```javascript
const ws = new WebSocket('ws://localhost:8000/ws/processing/550e8400-e29b-41d4-a716-446655440000');
```

**Messages:**
```json
// Progress update
{
  "type": "progress",
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "current_file": "exam_2023.pdf",
  "processed_files": 2,
  "total_files": 5,
  "percentage": 40
}

// File completed
{
  "type": "file_completed",
  "filename": "exam_2023.pdf",
  "questions_extracted": 25,
  "processing_time_ms": 3500
}

// Job completed
{
  "type": "job_completed",
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "total_questions": 125,
  "duration_seconds": 45
}

// Error
{
  "type": "error",
  "filename": "corrupted.pdf",
  "error": "Failed to process PDF: Invalid format"
}
```

## Error Responses

### Standard Error Format
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
  "request_id": "req_123456",
  "timestamp": "2024-01-01T12:00:00Z"
}
```

### Error Codes
- `VALIDATION_ERROR` - Invalid request data
- `NOT_FOUND` - Resource not found
- `PROCESSING_ERROR` - PDF processing failed
- `API_ERROR` - External API error
- `DATABASE_ERROR` - Database operation failed
- `RATE_LIMIT` - Too many requests

### HTTP Status Codes
- `200` - Success
- `201` - Created
- `400` - Bad Request
- `404` - Not Found
- `422` - Validation Error
- `429` - Too Many Requests
- `500` - Internal Server Error

## Rate Limiting

### Limits
- Upload: 10 requests per minute
- API calls: 100 requests per minute
- Export: 5 requests per minute

### Headers
```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1609459200
```

## Request/Response Examples

### Complete Upload to Save Flow

1. **Upload PDFs**
```bash
curl -X POST http://localhost:8000/api/upload \
  -F "pdfs=@exam1.pdf" \
  -F "pdfs=@exam2.pdf"
```

2. **Monitor Progress via WebSocket**
```javascript
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  updateProgressBar(data.percentage);
};
```

3. **Review Questions**
```bash
curl http://localhost:8000/api/questions?page=1&per_page=20
```

4. **Update Question**
```bash
curl -X PUT http://localhost:8000/api/questions/1 \
  -H "Content-Type: application/json" \
  -d '{"status": "approved", "marks": 10}'
```

5. **Save Approved**
```bash
curl -X POST http://localhost:8000/api/questions/save \
  -H "Content-Type: application/json" \
  -d '{"question_ids": [1, 2, 3]}'
```

## Pydantic Schemas

### Request Schemas
```python
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class QuestionUpdate(BaseModel):
    question_number: Optional[str]
    marks: Optional[int]
    year: Optional[str]
    level: Optional[str]
    topics: Optional[List[str]]
    question_type: Optional[str]
    question_text: Optional[str]
    status: Optional[str]

class BulkOperation(BaseModel):
    operation: str
    question_ids: List[int]
    data: dict

class SearchRequest(BaseModel):
    query: str
    limit: int = 10
    similarity_threshold: float = 0.8
    filters: Optional[dict] = None
```

### Response Schemas
```python
class QuestionResponse(BaseModel):
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

class PaginatedResponse(BaseModel):
    total: int
    page: int
    per_page: int
    total_pages: int
    questions: List[QuestionResponse]

class ProcessingStatus(BaseModel):
    job_id: str
    status: str
    total_files: int
    processed_files: int
    total_questions: int
    files: List[dict]
```

## API Documentation

FastAPI automatically generates OpenAPI documentation:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

## Security Considerations

1. **Input Validation**: All inputs validated with Pydantic
2. **SQL Injection Protection**: Parameterized queries via SQLAlchemy
3. **File Upload Limits**: Max 50MB per file, 1000 pages
4. **Rate Limiting**: Prevent abuse and API exhaustion
5. **CORS**: Configured for local development only
6. **Environment Variables**: Sensitive data never in code