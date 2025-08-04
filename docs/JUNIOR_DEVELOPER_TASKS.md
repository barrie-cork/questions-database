# Junior Developer Task List - PDF Question Extractor

## Overview
This document provides a step-by-step guide for junior developers to implement the PDF Question Extractor project. Each task is broken down into manageable steps with clear objectives and acceptance criteria.

## Prerequisites
- Python 3.11+ installed
- PostgreSQL 16+ with pgvector extension
- VS Code or similar IDE
- Basic Python, REST API, and SQL knowledge
- API keys for Mistral and Google services

## Important Notes
- **Use google-genai SDK (v0.9.0)**, NOT google-generativeai
- Follow async-first approach with FastAPI
- Implement proper error handling throughout
- Use 768-dimensional embeddings for efficiency

---

## Phase 1: Environment Setup (Day 1-2)

### Task 1.1: Project Setup ✅
**Objective**: Initialize project structure and virtual environment

**Steps**:
1. Clone the repository
2. Create and activate virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Copy `.env.example` to `.env` and add API keys

**Acceptance Criteria**:
- [ ] Virtual environment active
- [ ] All dependencies installed
- [ ] `.env` file configured
- [ ] Run `python test_setup.py` successfully

### Task 1.2: Database Setup ✅
**Objective**: Initialize PostgreSQL with pgvector

**Steps**:
1. Install PostgreSQL 16+ and pgvector
2. Create database and user:
   ```sql
   CREATE USER questionuser WITH PASSWORD 'your_password';
   CREATE DATABASE question_bank;
   GRANT ALL PRIVILEGES ON DATABASE question_bank TO questionuser;
   ```
3. Create `database/init_db.py`:
   ```python
   import asyncio
   import asyncpg
   from config import Config
   
   async def init_database():
       conn = await asyncpg.connect(
           host=Config.POSTGRES_HOST,
           port=Config.POSTGRES_PORT,
           user=Config.POSTGRES_USER,
           password=Config.POSTGRES_PASSWORD,
           database=Config.POSTGRES_DB
       )
       
       with open('database/schema.sql', 'r') as f:
           schema = f.read()
       
       await conn.execute(schema)
       print("✅ Database initialized!")
       await conn.close()
   
   if __name__ == "__main__":
       asyncio.run(init_database())
   ```
4. Run: `python database/init_db.py`

**Acceptance Criteria**:
- [ ] PostgreSQL running
- [ ] Database created with pgvector
- [ ] Tables created successfully
- [ ] Can connect via psql

---

## Phase 2: Core Services (Day 3-5)

### Task 2.1: OCR Service 🔍
**Objective**: Implement Mistral OCR for PDF processing

**File**: `services/ocr_service.py`

**Implementation Guide**:
1. Import required libraries
2. Create MistralOCRService class
3. Implement async PDF processing
4. Add error handling

**Code Template**:
```python
from mistralai import Mistral
import base64
import os

class MistralOCRService:
    def __init__(self):
        self.client = Mistral(api_key=os.getenv('MISTRAL_API_KEY'))
        
    async def process_pdf(self, pdf_path: str) -> str:
        """Process PDF and return markdown text"""
        # Your implementation here
        pass
```

**Testing**:
Create `test_ocr.py` and verify OCR works with a sample PDF

**Acceptance Criteria**:
- [ ] Processes PDFs successfully
- [ ] Returns markdown formatted text
- [ ] Handles errors gracefully

### Task 2.2: LLM Service with google-genai 🤖
**Objective**: Extract questions using Gemini 2.5 Flash Lite

**File**: `services/llm_service.py`

**Key Points**:
- Use `from google import genai` (NOT google.generativeai)
- Implement structured output with Pydantic
- Use async methods

**Code Structure**:
```python
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from typing import List
from enum import Enum

class QuestionType(str, Enum):
    MCQ = "MCQ"
    ESSAY = "Essay"
    SHORT_ANSWER = "Short Answer"
    NUMERICAL = "Numerical"
    TRUE_FALSE = "True/False"

class ExtractedQuestion(BaseModel):
    question_number: str = Field(..., description="Question number")
    marks: int = Field(..., description="Marks allocated")
    question_text: str = Field(..., description="Full question text")
    topics: List[str] = Field(..., description="Topics covered")
    question_type: QuestionType = Field(..., description="Type")

class GeminiLLMService:
    def __init__(self):
        self.client = genai.Client(api_key=os.getenv('GOOGLE_API_KEY'))
        self.model = "gemini-2.5-flash-lite-001"
    
    async def extract_questions(self, markdown_text: str, filename: str):
        # Implementation with structured output
        pass
```

**Acceptance Criteria**:
- [ ] Extracts questions from markdown
- [ ] Returns structured data
- [ ] Handles various question formats

### Task 2.3: Embedding Service 📊
**Objective**: Generate 768-dimensional embeddings

**File**: `services/embedding_service.py`

**Implementation Focus**:
- Use `models/embedding-001` (with prefix)
- Configure for 768 dimensions
- Implement batch processing with rate limiting

**Template**:
```python
from google import genai
from google.genai import types

class GeminiEmbeddingService:
    def __init__(self):
        self.client = genai.Client(api_key=os.getenv('GOOGLE_API_KEY'))
        self.model = "models/embedding-001"
        self.dimension = 768
        
    async def generate_embedding(self, text: str) -> List[float]:
        config = types.EmbedContentConfig(
            task_type="RETRIEVAL_DOCUMENT",
            output_dimensionality=self.dimension
        )
        # Complete implementation
        pass
```

**Acceptance Criteria**:
- [ ] Generates 768-dim embeddings
- [ ] Handles batch processing
- [ ] Respects rate limits

---

## Phase 3: Database Layer (Day 6-7)

### Task 3.1: Database Models 🗄️
**Objective**: Create SQLAlchemy models

**File**: `database/models.py`

**Requirements**:
- Use SQLAlchemy 2.0 with async support
- Include pgvector field type
- Three tables: extracted_questions, questions, question_embeddings

**Key Model**:
```python
from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, Integer, String, Text, ARRAY

class Question(Base):
    __tablename__ = 'questions'
    
    id = Column(Integer, primary_key=True)
    question_text = Column(Text, nullable=False)
    topics = Column(ARRAY(Text))
    # Add other fields
```

### Task 3.2: Database Operations 🔧
**Objective**: Implement CRUD and vector search

**File**: `database/operations.py`

**Key Methods**:
1. `store_extracted_questions()` - Save to temporary table
2. `get_pending_questions()` - Retrieve for review
3. `save_approved_questions()` - Move to permanent storage
4. `find_similar_questions()` - Vector similarity search

**Vector Search Example**:
```python
async def find_similar_questions(self, embedding: List[float], limit: int = 10):
    # Use pgvector's <=> operator for cosine distance
    query = text("""
        SELECT q.*, 1 - (qe.embedding <=> :embedding::vector(768)) AS similarity
        FROM questions q
        JOIN question_embeddings qe ON q.id = qe.question_id
        WHERE 1 - (qe.embedding <=> :embedding::vector(768)) > 0.7
        ORDER BY similarity DESC
        LIMIT :limit
    """)
    # Execute query
```

---

## Phase 4: API Development (Day 8-10)

### Task 4.1: FastAPI Setup 🚀
**Objective**: Create main application

**File**: `app.py`

**Structure**:
```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup code
    yield
    # Shutdown code

app = FastAPI(title="PDF Question Extractor", lifespan=lifespan)

# Add middleware
app.add_middleware(CORSMiddleware, allow_origins=["*"])

# Include routers
app.include_router(upload.router)
app.include_router(questions.router)
```

### Task 4.2: Upload Endpoints 📤
**Objective**: Handle PDF uploads

**File**: `api/routes/upload.py`

**Endpoints**:
- `POST /api/upload/pdf` - Single file
- `POST /api/upload/batch` - Multiple files

**Implementation Tips**:
- Use BackgroundTasks for processing
- Validate file types
- Return upload status

### Task 4.3: Question Endpoints 📝
**Objective**: CRUD operations for questions

**File**: `api/routes/questions.py`

**Required Endpoints**:
```python
GET    /api/questions/pending     # Get questions for review
PUT    /api/questions/{id}        # Update question
POST   /api/questions/{id}/approve # Approve question
POST   /api/questions/bulk-approve # Approve multiple
POST   /api/questions/save-approved # Save to permanent DB
POST   /api/questions/search/similar # Semantic search
```

### Task 4.4: PDF Processing Pipeline 🔄
**Objective**: Integrate all services

**File**: `services/pdf_processor.py`

**Pipeline Steps**:
1. OCR → Markdown text
2. LLM → Extract questions
3. Store in temporary DB
4. Generate embeddings
5. Return processing status

**Error Handling**:
- Wrap each step in try/except
- Log errors appropriately
- Return meaningful error messages

---

## Phase 5: Frontend (Day 11-12)

### Task 5.1: HTML Interface 🎨
**Objective**: Create user interface

**File**: `static/index.html`

**Sections**:
1. Upload area with drag-drop
2. Question review table
3. Semantic search interface

### Task 5.2: JavaScript Implementation 💻
**Objective**: Add interactivity

**File**: `static/js/app.js`

**Key Features**:
- Tabulator.js for data grid
- Auto-save on edit
- Bulk operations
- Real-time filtering

**Tabulator Setup**:
```javascript
questionsTable = new Tabulator("#questions-table", {
    height: "600px",
    pagination: "local",
    paginationSize: 20,
    columns: [
        {title: "✓", field: "approved", formatter: "tickCross"},
        {title: "Question", field: "question_text", editor: "textarea"},
        // Add other columns
    ]
});
```

---

## Phase 6: Testing & Deployment (Day 13-14)

### Task 6.1: Unit Tests 🧪
**Objective**: Test core services

**File**: `tests/test_services.py`

**Test Coverage**:
- OCR service
- LLM extraction
- Embedding generation
- Database operations

### Task 6.2: Docker Setup 🐳
**Objective**: Containerize application

**Files**:
- `Dockerfile`
- `docker-compose.yml`

**Key Services**:
```yaml
services:
  postgres:
    image: pgvector/pgvector:pg16
    # Configuration
    
  app:
    build: .
    depends_on:
      - postgres
    # Configuration
```

### Task 6.3: Health Checks 🏥
**Objective**: Monitoring endpoints

**Implementation**:
```python
@router.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "services": {
            "database": await check_db(),
            "ocr": check_api("MISTRAL_API_KEY"),
            "llm": check_api("GOOGLE_API_KEY")
        }
    }
```

---

## Best Practices Checklist

### Code Quality
- [ ] Use type hints everywhere
- [ ] Add docstrings to functions
- [ ] Handle exceptions properly
- [ ] Log important events

### Performance
- [ ] Use async/await throughout
- [ ] Batch API calls when possible
- [ ] Implement rate limiting
- [ ] Cache embeddings

### Security
- [ ] Never commit API keys
- [ ] Validate all inputs
- [ ] Use environment variables
- [ ] Implement CORS properly

### Testing
- [ ] Write unit tests
- [ ] Test error scenarios
- [ ] Verify rate limiting
- [ ] Check memory usage

---

## Common Issues & Solutions

### Issue: "ModuleNotFoundError: No module named 'google.generativeai'"
**Solution**: You're using the old SDK. Use:
```python
from google import genai  # Correct
# NOT: import google.generativeai
```

### Issue: "Embedding dimension mismatch"
**Solution**: Ensure you're using 768 dimensions:
```python
config = types.EmbedContentConfig(
    output_dimensionality=768  # Must be 768
)
```

### Issue: "Rate limit exceeded"
**Solution**: Implement delay between requests:
```python
await asyncio.sleep(0.1)  # Between API calls
```

### Issue: "pgvector not found"
**Solution**: Install pgvector extension:
```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

---

## Resources

### Documentation
- [google-genai SDK](https://googleapis.github.io/python-genai/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [pgvector](https://github.com/pgvector/pgvector)
- [Tabulator.js](http://tabulator.info/)

### Example Code
- Check `/docs/gemini_api_implementation_guide_2025.md` for detailed examples
- Review PRD.md for requirements

### Support
- Create issues in project repository
- Check logs in `./logs/` directory
- Use health endpoints for debugging

---

## Final Checklist

Before marking project complete:
- [ ] All services working end-to-end
- [ ] Can upload and process PDFs
- [ ] Questions extracted correctly
- [ ] Review interface functional
- [ ] Semantic search working
- [ ] Tests passing
- [ ] Docker deployment successful
- [ ] Documentation complete

Good luck! 🚀