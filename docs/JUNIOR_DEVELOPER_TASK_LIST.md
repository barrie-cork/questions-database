# Junior Developer Task List - PDF Question Extractor

## Overview
This document provides a comprehensive, step-by-step task list for junior developers to implement the PDF Question Extractor project. Each task is broken down into manageable chunks with clear instructions and expected outcomes.

## Prerequisites
- Python 3.11+ installed
- PostgreSQL 16+ installed (or Docker Desktop for containerized PostgreSQL)
- VS Code or similar IDE
- Basic understanding of Python, APIs, and databases
- Git installed for version control

---

## Phase 1: Environment Setup (Day 1)

### Task 1.1: Verify Python Environment
**Time Estimate**: 15 minutes
```bash
# Check Python version (should be 3.11 or higher)
python --version

# Navigate to project directory
cd /mnt/d/Python/Projects/Dave/questions_pdf_to_sheet/pdf_question_extractor

# Activate virtual environment (already created)
source ../super_c/bin/activate  # On Windows: ..\super_c\Scripts\activate

# Verify activation
which python  # Should show path to virtual environment
```

### Task 1.2: Install Dependencies
**Time Estimate**: 30 minutes
```bash
# Install all dependencies
pip install -r requirements.txt

# If you encounter errors, install one by one:
pip install fastapi==0.115.0
pip install uvicorn[standard]==0.34.0
pip install google-genai==0.9.0
pip install mistralai==1.2.3
# ... continue for all dependencies

# Verify installation
python test_setup.py
```

### Task 1.3: Configure Environment Variables
**Time Estimate**: 10 minutes
```bash
# Check .env file exists
ls -la .env

# Verify all API keys are set
python -c "from dotenv import load_dotenv; import os; load_dotenv(); print('Keys configured:', all([os.getenv('MISTRAL_API_KEY'), os.getenv('GOOGLE_API_KEY')]))"
```

---

## Phase 2: Database Setup (Day 1-2)

### Task 2.1: Install PostgreSQL with pgvector
**Time Estimate**: 45 minutes

**Option A: Using Docker (Recommended)**
```bash
# Create docker-compose.yml for PostgreSQL
cat > docker-compose.yml << 'EOF'
version: '3.8'
services:
  postgres:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_USER: questionuser
      POSTGRES_PASSWORD: your_secure_password_here
      POSTGRES_DB: question_bank
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  postgres_data:
EOF

# Start PostgreSQL
docker-compose up -d

# Verify it's running
docker ps
```

**Option B: Local Installation**
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install postgresql-16 postgresql-16-pgvector

# macOS
brew install postgresql@16
brew install pgvector
```

### Task 2.2: Create Database Schema
**Time Estimate**: 30 minutes

1. Create a file `database/init_db.py`:
```python
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import os
from dotenv import load_dotenv

load_dotenv()

def init_database():
    # Connection parameters
    conn_params = {
        'host': os.getenv('POSTGRES_HOST', 'localhost'),
        'port': os.getenv('POSTGRES_PORT', '5432'),
        'user': os.getenv('POSTGRES_USER', 'questionuser'),
        'password': os.getenv('POSTGRES_PASSWORD', 'your_secure_password_here')
    }
    
    # Connect to PostgreSQL
    conn = psycopg2.connect(**conn_params, database='postgres')
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cur = conn.cursor()
    
    # Create database if not exists
    cur.execute("SELECT 1 FROM pg_database WHERE datname = 'question_bank'")
    if not cur.fetchone():
        cur.execute("CREATE DATABASE question_bank")
        print("Database created successfully")
    
    cur.close()
    conn.close()
    
    # Connect to the new database
    conn = psycopg2.connect(**conn_params, database='question_bank')
    cur = conn.cursor()
    
    # Create extensions
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
    cur.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
    print("Extensions created successfully")
    
    # Execute schema.sql
    with open('database/schema.sql', 'r') as f:
        cur.execute(f.read())
    
    conn.commit()
    print("Schema created successfully")
    
    cur.close()
    conn.close()

if __name__ == "__main__":
    init_database()
```

2. Run the initialization:
```bash
python database/init_db.py
```

### Task 2.3: Verify Database Setup
**Time Estimate**: 15 minutes
```bash
# Connect to database
psql -U questionuser -d question_bank -h localhost

# In psql prompt, verify tables:
\dt
# Should show: extracted_questions, questions, question_embeddings

# Check extensions:
\dx
# Should show: plpgsql, pg_trgm, vector

# Exit psql
\q
```

---

## Phase 3: Implement Core Services (Day 2-3)

### Task 3.1: Create OCR Service
**Time Estimate**: 1 hour

1. Create `services/__init__.py`:
```python
"""Services module for PDF Question Extractor"""
```

2. Copy the OCR service from SERVICE_IMPLEMENTATION_SPEC.md to `services/ocr_service.py`
   - The complete code is already provided in the spec
   - Make sure to include all imports and the full MistralOCRService class

3. Test the OCR service:
```python
# create test_ocr.py
import asyncio
from services.ocr_service import MistralOCRService
import os
from dotenv import load_dotenv

load_dotenv()

async def test_ocr():
    ocr = MistralOCRService(os.getenv('MISTRAL_API_KEY'))
    # Use a small test PDF
    result = await ocr.process_pdf('test_files/sample.pdf')
    print(f"OCR Result (first 500 chars): {result[:500]}")

if __name__ == "__main__":
    asyncio.run(test_ocr())
```

### Task 3.2: Create LLM Service
**Time Estimate**: 1.5 hours

1. Copy the LLM service from SERVICE_IMPLEMENTATION_SPEC.md to `services/llm_service.py`
   - Include all Pydantic models
   - Include the RateLimiter class
   - Include the complete GeminiLLMService class

2. Create a simple test:
```python
# create test_llm.py
import asyncio
from services.llm_service import GeminiLLMService
import os
from dotenv import load_dotenv

load_dotenv()

async def test_llm():
    llm = GeminiLLMService(os.getenv('GOOGLE_API_KEY'))
    
    # Test with sample OCR text
    sample_text = """
    Mathematics Exam 2024
    Level: High School
    
    Question 1 (5 marks)
    Calculate the area of a circle with radius 7cm.
    
    Question 2 (10 marks)
    Explain the difference between mean, median, and mode.
    """
    
    result = await llm.extract_questions(sample_text, "test.pdf")
    print(f"Extracted {len(result.questions)} questions")
    for q in result.questions:
        print(f"- Q{q.question_number}: {q.question_text[:50]}... ({q.marks} marks)")

if __name__ == "__main__":
    asyncio.run(test_llm())
```

### Task 3.3: Create Embedding Service
**Time Estimate**: 1 hour

1. Copy the embedding service from SERVICE_IMPLEMENTATION_SPEC.md to `services/embedding_service.py`
   - Include the complete GeminiEmbeddingService class
   - Make sure the RateLimiter is imported from llm_service

2. Test the embedding service:
```python
# create test_embedding.py
import asyncio
from services.embedding_service import GeminiEmbeddingService
import os
from dotenv import load_dotenv

load_dotenv()

async def test_embedding():
    embedding_service = GeminiEmbeddingService(os.getenv('GOOGLE_API_KEY'))
    
    # Test single embedding
    text = "What is the capital of France?"
    embedding = await embedding_service.generate_embedding(text)
    print(f"Embedding dimensions: {len(embedding)}")
    print(f"First 5 values: {embedding[:5]}")
    
    # Test search embedding
    search_embedding = await embedding_service.search_embedding("geography capitals")
    print(f"Search embedding dimensions: {len(search_embedding)}")

if __name__ == "__main__":
    asyncio.run(test_embedding())
```

### Task 3.4: Create PDF Processor
**Time Estimate**: 1 hour

1. Copy the PDF processor from SERVICE_IMPLEMENTATION_SPEC.md to `services/pdf_processor.py`
2. Create test files directory:
```bash
mkdir -p test_files
# Add a sample PDF to test_files/sample.pdf
```

---

## Phase 4: Implement Database Operations (Day 3)

### Task 4.1: Create Database Models
**Time Estimate**: 45 minutes

1. Ensure `database/models.py` has all SQLAlchemy models matching the schema
2. Create `database/session.py`:
```python
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base
import os
from dotenv import load_dotenv

load_dotenv()

# Create async engine
DATABASE_URL = (
    f"postgresql+asyncpg://{os.getenv('POSTGRES_USER')}:"
    f"{os.getenv('POSTGRES_PASSWORD')}@"
    f"{os.getenv('POSTGRES_HOST')}:"
    f"{os.getenv('POSTGRES_PORT')}/"
    f"{os.getenv('POSTGRES_DB')}"
)

engine = create_async_engine(DATABASE_URL, echo=True)
AsyncSessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

Base = declarative_base()

async def get_db():
    async with AsyncSessionLocal() as session:
        yield session
```

### Task 4.2: Create Vector Operations
**Time Estimate**: 1 hour

Create `database/vector_operations.py`:
```python
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
import json
from typing import List, Dict, Optional

class VectorDatabaseOperations:
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def store_question_with_embedding(
        self, 
        question: Dict, 
        embedding: List[float]
    ) -> int:
        """Store question and embedding atomically"""
        
        # First, store in extracted_questions for review
        result = await self.session.execute(
            text("""
                INSERT INTO extracted_questions (
                    question_number, marks, year, level, topics,
                    question_type, question_text, source_pdf, status
                ) VALUES (
                    :q_num, :marks, :year, :level, :topics::text[],
                    :q_type, :q_text, :source, 'pending'
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
        await self.session.commit()
        return question_id
    
    async def approve_questions(self, question_ids: List[int]):
        """Move approved questions to permanent storage"""
        # Implementation to move from extracted_questions to questions table
        pass
```

---

## Phase 5: Build FastAPI Application (Day 4)

### Task 5.1: Create Main Application
**Time Estimate**: 1 hour

Create `app.py`:
```python
from fastapi import FastAPI, HTTPException, UploadFile, File, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
from dotenv import load_dotenv

load_dotenv()

# Create FastAPI app
app = FastAPI(
    title="PDF Question Extractor",
    description="Extract questions from exam PDFs using AI",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "1.0.0"}

# Root endpoint
@app.get("/")
async def read_root():
    return {"message": "PDF Question Extractor API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
```

### Task 5.2: Create Upload Endpoint
**Time Estimate**: 1 hour

Add to `app.py`:
```python
from services.pdf_processor import PDFQuestionProcessor
from services.ocr_service import MistralOCRService
from services.llm_service import GeminiLLMService
from services.embedding_service import GeminiEmbeddingService
import aiofiles
from pathlib import Path

# Initialize services
ocr_service = MistralOCRService(os.getenv('MISTRAL_API_KEY'))
llm_service = GeminiLLMService(os.getenv('GOOGLE_API_KEY'))
embedding_service = GeminiEmbeddingService(os.getenv('GOOGLE_API_KEY'))
processor = PDFQuestionProcessor(ocr_service, llm_service, embedding_service)

# Create upload directory
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

@app.post("/api/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload and process a PDF file"""
    
    # Validate file type
    if not file.filename.endswith('.pdf'):
        raise HTTPException(400, "Only PDF files are allowed")
    
    # Save file
    file_path = UPLOAD_DIR / file.filename
    async with aiofiles.open(file_path, 'wb') as f:
        content = await file.read()
        await f.write(content)
    
    try:
        # Process PDF
        result = await processor.process_pdf(str(file_path), file.filename)
        
        # Store in database (implement this)
        # ...
        
        return {
            "status": "success",
            "filename": file.filename,
            "questions_extracted": len(result['questions']),
            "processing_time": result['processing_time']
        }
    
    except Exception as e:
        raise HTTPException(500, f"Processing failed: {str(e)}")
    
    finally:
        # Cleanup
        if file_path.exists():
            file_path.unlink()
```

### Task 5.3: Create Question Management Endpoints
**Time Estimate**: 1.5 hours

Create `api/routes/questions.py`:
```python
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from database.session import get_db
from typing import List, Optional

router = APIRouter(prefix="/api/questions", tags=["questions"])

@router.get("/")
async def get_questions(
    status: Optional[str] = None,
    source_pdf: Optional[str] = None,
    limit: int = 20,
    offset: int = 0,
    db: AsyncSession = Depends(get_db)
):
    """Get paginated list of extracted questions"""
    # Implement query logic
    pass

@router.put("/{question_id}")
async def update_question(
    question_id: int,
    question_data: dict,
    db: AsyncSession = Depends(get_db)
):
    """Update a single question"""
    # Implement update logic
    pass

@router.post("/approve")
async def approve_questions(
    question_ids: List[int],
    db: AsyncSession = Depends(get_db)
):
    """Approve questions and move to permanent storage"""
    # Implement approval logic
    pass
```

---

## Phase 6: Create Web UI (Day 5)

### Task 6.1: Create HTML Structure
**Time Estimate**: 45 minutes

Copy the HTML from PRD.md to `static/index.html` and ensure all paths are correct.

### Task 6.2: Create JavaScript Application
**Time Estimate**: 1 hour

Copy the JavaScript from PRD.md to `static/js/app.js` and test the Tabulator integration.

### Task 6.3: Create CSS Styling
**Time Estimate**: 30 minutes

Create `static/css/style.css`:
```css
body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    margin: 0;
    padding: 0;
    background-color: #f5f5f5;
}

.container {
    max-width: 1400px;
    margin: 0 auto;
    padding: 20px;
}

.upload-section {
    background: white;
    padding: 20px;
    border-radius: 8px;
    margin-bottom: 20px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.review-section {
    background: white;
    padding: 20px;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.controls {
    margin-bottom: 20px;
    display: flex;
    gap: 10px;
}

.controls button {
    padding: 8px 16px;
    border: none;
    border-radius: 4px;
    background: #007bff;
    color: white;
    cursor: pointer;
}

.controls button:hover {
    background: #0056b3;
}

.save-btn {
    background: #28a745 !important;
}

.save-btn:hover {
    background: #218838 !important;
}

.status-bar {
    margin-top: 20px;
    display: flex;
    justify-content: space-between;
    color: #666;
}

#uploadProgress {
    margin-top: 10px;
    font-size: 14px;
    color: #666;
}

.modified {
    background-color: #fff3cd !important;
}
```

---

## Phase 7: Testing and Validation (Day 5-6)

### Task 7.1: Create Integration Tests
**Time Estimate**: 2 hours

Create `tests/test_integration.py`:
```python
import pytest
import asyncio
from services.pdf_processor import PDFQuestionProcessor
from services.ocr_service import MistralOCRService
from services.llm_service import GeminiLLMService
from services.embedding_service import GeminiEmbeddingService
import os

@pytest.fixture
def processor():
    ocr = MistralOCRService(os.getenv('MISTRAL_API_KEY'))
    llm = GeminiLLMService(os.getenv('GOOGLE_API_KEY'))
    embedding = GeminiEmbeddingService(os.getenv('GOOGLE_API_KEY'))
    return PDFQuestionProcessor(ocr, llm, embedding)

@pytest.mark.asyncio
async def test_pdf_processing(processor):
    # Test with a sample PDF
    result = await processor.process_pdf('test_files/sample.pdf', 'sample.pdf')
    
    assert 'questions' in result
    assert len(result['questions']) > 0
    assert 'exam_info' in result
```

### Task 7.2: Create API Tests
**Time Estimate**: 1 hour

Create `tests/test_api.py`:
```python
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_upload_endpoint():
    # Test file upload
    with open("test_files/sample.pdf", "rb") as f:
        response = client.post(
            "/api/upload",
            files={"file": ("sample.pdf", f, "application/pdf")}
        )
    
    assert response.status_code == 200
    assert "questions_extracted" in response.json()
```

### Task 7.3: Run Full System Test
**Time Estimate**: 1 hour

1. Start the application:
```bash
python app.py
```

2. Open browser to http://localhost:8000

3. Test the workflow:
   - Upload a PDF
   - Review extracted questions
   - Edit some questions
   - Approve/reject questions
   - Save to database
   - Verify in PostgreSQL

---

## Phase 8: Documentation and Deployment (Day 6)

### Task 8.1: Create README
**Time Estimate**: 30 minutes

Create a comprehensive README.md with:
- Project overview
- Installation instructions
- Configuration details
- Usage examples
- API documentation

### Task 8.2: Create Docker Configuration
**Time Estimate**: 45 minutes

Create `Dockerfile`:
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create directories
RUN mkdir -p uploads cache logs

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Task 8.3: Final Testing Checklist
**Time Estimate**: 1 hour

- [ ] All API endpoints working
- [ ] PDF upload and processing successful
- [ ] Questions extracted correctly
- [ ] Embeddings generated
- [ ] Database storage working
- [ ] UI displays questions
- [ ] Edit functionality working
- [ ] Approval process working
- [ ] Export to CSV working
- [ ] Error handling tested

---

## Troubleshooting Guide

### Common Issues

1. **API Key Errors**
   - Verify all keys in .env file
   - Check key permissions and quotas

2. **Database Connection Issues**
   - Verify PostgreSQL is running
   - Check connection parameters
   - Ensure pgvector extension is installed

3. **Memory Issues with Large PDFs**
   - Implement streaming for large files
   - Increase Docker memory limits
   - Use chunking for processing

4. **Rate Limiting Errors**
   - Reduce concurrent requests
   - Implement exponential backoff
   - Check API quotas

---

## Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Google GenAI SDK Docs](https://googleapis.github.io/python-genai/)
- [Mistral Documentation](https://docs.mistral.ai/)
- [pgvector Documentation](https://github.com/pgvector/pgvector)
- [Tabulator Documentation](http://tabulator.info/)

---

## Success Criteria

The project is complete when:
1. ✅ PDFs can be uploaded and processed
2. ✅ Questions are extracted with >90% accuracy
3. ✅ All metadata is captured correctly
4. ✅ UI allows review and editing
5. ✅ Approved questions are saved to database
6. ✅ Vector search is functional
7. ✅ System handles errors gracefully
8. ✅ Performance is acceptable (<30s per PDF)
9. ✅ All tests pass
10. ✅ Documentation is complete