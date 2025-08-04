# PDF Question Extractor - Junior Developer Task List

This guide breaks down the project into small, manageable tasks. Each task should take 1-4 hours to complete.

## 📋 How to Use This Guide

1. Complete tasks in order - they build on each other
2. Test each task before moving to the next
3. Ask for help if stuck for more than 30 minutes
4. Commit your code after each completed section
5. Check off tasks as you complete them

---

## 🚀 Phase 1: Environment Setup (Day 1)

### Task 1.1: Install Required Software
**Time: 1-2 hours**

#### Option A: Docker Setup (Recommended)
- [ ] Install Docker Desktop
  - Windows: Download from docker.com/products/docker-desktop
  - Mac: `brew install --cask docker`
  - Linux: Follow docs.docker.com/engine/install
- [ ] Install Docker Compose (usually included with Docker Desktop)
- [ ] Install VS Code or your preferred editor
- [ ] Install Git if not already installed

**Test**: Run `docker --version` and `docker-compose --version` to verify

#### Option B: Traditional Setup
- [ ] Install Python 3.11 or higher
  - Windows: Download from python.org
  - Mac: `brew install python@3.11`
  - Linux: `sudo apt install python3.11`
- [ ] Install PostgreSQL 16 with pgvector
  - Windows: Download installer from postgresql.org
  - Mac: `brew install postgresql@16 && brew install pgvector`
  - Linux: `sudo apt install postgresql-16 postgresql-16-pgvector`
- [ ] Install Poetry for dependency management
  ```bash
  curl -sSL https://install.python-poetry.org | python3 -
  ```

**Test**: Run `python --version`, `psql --version`, and `poetry --version` to verify

### Task 1.2: Setup Project Environment
**Time: 30 minutes**

#### Option A: Docker Setup with super_c Environment (Recommended)
```bash
# Navigate to Dave's projects directory
cd /mnt/d/Python/Projects/Dave

# Activate the existing super_c virtual environment
source super_c/bin/activate

# Navigate to project
cd questions_pdf_to_sheet

# Copy environment variables
cp .env.example .env

# Build and start containers (Python runs inside container, but we use super_c for Docker commands)
docker-compose up -d --build

# Check if services are running
docker-compose ps
```

- [ ] super_c virtual environment activated
- [ ] `.env` file created and configured
- [ ] Docker images built successfully
- [ ] Both app and postgres containers running
- [ ] No error messages during startup

**Test**: 
- Visit http://localhost:8000/docs to see API documentation
- Run `docker-compose logs app` to check application logs

#### Option B: Traditional Setup with super_c
```bash
# Navigate to Dave's projects directory
cd /mnt/d/Python/Projects/Dave

# Activate the existing super_c virtual environment
source super_c/bin/activate

# Navigate to project directory
cd questions_pdf_to_sheet/pdf_question_extractor

# Install additional dependencies if needed
pip install -r requirements.txt
```

- [ ] super_c virtual environment activated
- [ ] All packages installed successfully
- [ ] No error messages during installation

**Test**: Run `pip list` to see all installed packages

### Task 1.3: Configure Environment Variables
**Time: 30 minutes**

- [ ] Copy `.env.example` to `.env`
  ```bash
  cp .env.example .env
  ```
- [ ] Get API keys:
  - [ ] Mistral API key from https://console.mistral.ai/
  - [ ] Google API key from https://makersuite.google.com/app/apikey
- [ ] Edit `.env` file with your keys:
  ```
  MISTRAL_API_KEY=your_actual_mistral_key_here
  GOOGLE_API_KEY=your_actual_google_key_here
  ```
- [ ] Generate a secure secret key:
  ```python
  import secrets
  print(secrets.token_hex(32))
  ```
- [ ] Add the secret key to `.env`

**Test**: Create `test_env.py`:
```python
from config import Config
print("Mistral Key exists:", bool(Config.MISTRAL_API_KEY))
print("Google Key exists:", bool(Config.GOOGLE_API_KEY))
```

---

## 🗄️ Phase 2: Database Setup (Day 1-2)

### Task 2.1: Create PostgreSQL Database
**Time: 1 hour**

#### Option A: Docker Setup (Recommended)
The database is automatically created when you run `docker-compose up`. To verify:

```bash
# Check database is running
docker-compose ps

# Connect to database
docker-compose exec postgres psql -U postgres -d question_bank

# Or use the Makefile
make db-shell
```

- [ ] PostgreSQL container is running
- [ ] Can connect to database
- [ ] pgvector extension is installed (run `\dx` in psql)

#### Option B: Traditional Setup
- [ ] Start PostgreSQL service:
  - Windows: Check Services app
  - Mac: `brew services start postgresql@16`
  - Linux: `sudo systemctl start postgresql`

- [ ] Create database and user:
  ```bash
  # Connect to PostgreSQL as superuser
  sudo -u postgres psql
  
  # Run these SQL commands:
  CREATE DATABASE question_bank;
  CREATE USER questionuser WITH PASSWORD 'your_secure_password';
  GRANT ALL PRIVILEGES ON DATABASE question_bank TO questionuser;
  \q
  ```

- [ ] Update `.env` with your database password

**Test**: Try connecting:
```bash
psql -U questionuser -d question_bank -h localhost
```

### Task 2.2: Install pgvector Extension
**Time: 30 minutes**

- [ ] Install pgvector:
  - Ubuntu: `sudo apt install postgresql-16-pgvector`
  - Mac: `brew install pgvector`
  - Windows: Follow [pgvector guide](https://github.com/pgvector/pgvector#installation)

- [ ] Enable extension in database:
  ```bash
  psql -U questionuser -d question_bank
  CREATE EXTENSION vector;
  \q
  ```

**Test**: Check extension:
```sql
SELECT * FROM pg_extension WHERE extname = 'vector';
```

### Task 2.3: Initialize Database Schema
**Time: 30 minutes**

- [ ] Run the initialization script:
  ```bash
  python database/init_db.py
  ```

- [ ] Verify all tables were created
- [ ] Check for any error messages

**Test**: List all tables:
```bash
psql -U questionuser -d question_bank -c "\dt"
```

---

## 🔧 Phase 3: Core Services Implementation (Days 3-5)

### Task 3.1: Create Basic Service Structure
**Time: 1 hour**

Create `services/__init__.py`:
```python
"""Service modules for PDF Question Extractor"""
```

- [ ] File created
- [ ] Can import from services module

### Task 3.2: Implement OCR Service - Part 1 (Basic Structure)
**Time: 2 hours**

Create `services/ocr_service.py`:
```python
"""Mistral OCR Service for PDF processing"""
import os
from mistralai import Mistral
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class OCRService:
    """Service for OCR processing using Mistral API"""
    
    def __init__(self, api_key: str = None):
        """Initialize OCR service with API key"""
        self.api_key = api_key or os.getenv('MISTRAL_API_KEY')
        if not self.api_key:
            raise ValueError("MISTRAL_API_KEY is required")
        
        # Initialize Mistral client
        self.client = Mistral(api_key=self.api_key)
        logger.info("OCR Service initialized")
    
    def process_pdf(self, pdf_path: str) -> str:
        """Process a single PDF file and return markdown text"""
        # TODO: Implement in next task
        pass
```

- [ ] Create the file
- [ ] Add imports
- [ ] Create OCRService class
- [ ] Add __init__ method

**Test**: 
```python
from services.ocr_service import OCRService
ocr = OCRService()  # Should work if API key is in .env
```

### Task 3.3: Implement OCR Service - Part 2 (PDF Processing)
**Time: 2 hours**

Add to `services/ocr_service.py`:
```python
import base64
from pathlib import Path

def process_pdf(self, pdf_path: str) -> str:
    """Process a single PDF file and return markdown text"""
    try:
        logger.info(f"Processing PDF: {pdf_path}")
        
        # Read PDF file
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        # Check file size (max 50MB)
        file_size_mb = pdf_path.stat().st_size / (1024 * 1024)
        if file_size_mb > 50:
            raise ValueError(f"PDF too large: {file_size_mb:.1f}MB (max 50MB)")
        
        # Read file as base64
        with open(pdf_path, 'rb') as f:
            pdf_base64 = base64.b64encode(f.read()).decode('utf-8')
        
        # Call Mistral OCR API
        response = self.client.ocr.process(
            model="mistral-ocr-latest",
            document={
                "type": "document_base64",
                "document_base64": pdf_base64
            }
        )
        
        # Extract markdown text from response
        markdown_text = response.content
        logger.info(f"OCR completed for {pdf_path.name}")
        
        return markdown_text
        
    except Exception as e:
        logger.error(f"OCR failed for {pdf_path}: {str(e)}")
        raise
```

- [ ] Add file reading logic
- [ ] Add size validation
- [ ] Add API call
- [ ] Add error handling

### Task 3.4: Create Test Script for OCR
**Time: 1 hour**

Create `test_ocr.py`:
```python
"""Test OCR service with a sample PDF"""
from services.ocr_service import OCRService
import logging

logging.basicConfig(level=logging.INFO)

def test_ocr():
    # Initialize service
    ocr = OCRService()
    
    # Test with a sample PDF (you need to provide one)
    pdf_path = "samples/test.pdf"  # Put a small test PDF here
    
    try:
        result = ocr.process_pdf(pdf_path)
        print(f"OCR Result (first 500 chars):\n{result[:500]}")
        print(f"\nTotal length: {len(result)} characters")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_ocr()
```

- [ ] Create test script
- [ ] Add a small test PDF to samples/ folder
- [ ] Run test and verify output

### Task 3.5: Implement LLM Service - Part 1 (Basic Structure)
**Time: 2 hours**

Create `services/llm_service.py`:
```python
"""Gemini LLM Service for question extraction"""
import os
import google.generativeai as genai
from typing import List, Dict, Any
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)


class Question(BaseModel):
    """Single question model"""
    question_number: str
    marks: int
    question_text: str
    topics: List[str]
    question_type: str


class ExamPaper(BaseModel):
    """Exam paper with extracted questions"""
    year: str
    level: str
    source_pdf: str
    questions: List[Question]


class LLMService:
    """Service for question extraction using Gemini"""
    
    def __init__(self, api_key: str = None):
        """Initialize LLM service"""
        self.api_key = api_key or os.getenv('GOOGLE_API_KEY')
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY is required")
        
        # Configure Gemini
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash-lite')
        logger.info("LLM Service initialized")
    
    def extract_questions(self, markdown_text: str, pdf_filename: str) -> ExamPaper:
        """Extract questions from markdown text"""
        # TODO: Implement in next task
        pass
```

- [ ] Create file with imports
- [ ] Define Pydantic models
- [ ] Create LLMService class
- [ ] Initialize Gemini

### Task 3.6: Implement LLM Service - Part 2 (Question Extraction)
**Time: 3 hours**

Add to `services/llm_service.py`:
```python
def extract_questions(self, markdown_text: str, pdf_filename: str) -> ExamPaper:
    """Extract questions from markdown text"""
    try:
        logger.info(f"Extracting questions from {pdf_filename}")
        
        # Create prompt
        prompt = f"""
        Extract all questions from this exam paper. For each question, identify:
        - Question number
        - Marks allocated
        - Full question text
        - Topics/subjects covered
        - Question type (MCQ, Essay, Short Answer, etc.)
        
        Also identify the year and level of the exam.
        
        Exam paper content:
        {markdown_text}
        
        Return as JSON matching this structure:
        {{
            "year": "2023",
            "level": "Grade 12",
            "source_pdf": "{pdf_filename}",
            "questions": [
                {{
                    "question_number": "1",
                    "marks": 5,
                    "question_text": "Full question text here",
                    "topics": ["topic1", "topic2"],
                    "question_type": "Essay"
                }}
            ]
        }}
        """
        
        # Generate response
        response = self.model.generate_content(
            prompt,
            generation_config={
                "response_mime_type": "application/json",
                "response_schema": ExamPaper.model_json_schema()
            }
        )
        
        # Parse response
        exam_paper = ExamPaper.model_validate_json(response.text)
        logger.info(f"Extracted {len(exam_paper.questions)} questions")
        
        return exam_paper
        
    except Exception as e:
        logger.error(f"Question extraction failed: {str(e)}")
        raise
```

- [ ] Create extraction prompt
- [ ] Add JSON schema configuration
- [ ] Parse and validate response
- [ ] Add error handling

### Task 3.7: Create Test Script for LLM
**Time: 1 hour**

Create `test_llm.py`:
```python
"""Test LLM service with sample text"""
from services.llm_service import LLMService
import logging

logging.basicConfig(level=logging.INFO)

def test_llm():
    # Initialize service
    llm = LLMService()
    
    # Sample exam text
    sample_text = """
    Mathematics Exam - Grade 12 - 2023
    
    Question 1 (5 marks)
    Solve the following equation: 2x + 3 = 15
    
    Question 2 (10 marks)
    Write an essay on the importance of calculus in real life.
    
    Question 3 (3 marks each)
    Multiple Choice:
    a) What is 5 + 5?
       A) 10  B) 11  C) 9  D) 8
    """
    
    try:
        result = llm.extract_questions(sample_text, "test.pdf")
        print(f"Year: {result.year}")
        print(f"Level: {result.level}")
        print(f"Questions found: {len(result.questions)}")
        
        for q in result.questions:
            print(f"\nQ{q.question_number}: {q.question_text[:50]}...")
            print(f"  Marks: {q.marks}, Type: {q.question_type}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_llm()
```

- [ ] Create test script
- [ ] Test with sample text
- [ ] Verify output format

### Task 3.8: Implement Embedding Service
**Time: 2 hours**

Create `services/embedding_service.py`:
```python
"""Gemini Embedding Service for semantic search"""
import os
import google.generativeai as genai
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for generating question embeddings"""
    
    def __init__(self, api_key: str = None):
        """Initialize embedding service"""
        self.api_key = api_key or os.getenv('GOOGLE_API_KEY')
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY is required")
        
        genai.configure(api_key=self.api_key)
        self.model_name = "gemini-embedding-001"
        logger.info("Embedding Service initialized")
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        try:
            # Use Gemini to generate embedding
            result = genai.embed_content(
                model=self.model_name,
                content=text
            )
            
            # Extract embedding vector
            embedding = result['embedding']
            logger.debug(f"Generated embedding of dimension {len(embedding)}")
            
            return embedding
            
        except Exception as e:
            logger.error(f"Embedding generation failed: {str(e)}")
            raise
    
    def generate_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        embeddings = []
        
        for i, text in enumerate(texts):
            try:
                embedding = self.generate_embedding(text)
                embeddings.append(embedding)
                
                if (i + 1) % 10 == 0:
                    logger.info(f"Generated {i + 1}/{len(texts)} embeddings")
                    
            except Exception as e:
                logger.error(f"Failed to embed text {i}: {str(e)}")
                embeddings.append(None)
        
        return embeddings
```

- [ ] Create embedding service
- [ ] Add single embedding method
- [ ] Add batch embedding method
- [ ] Add progress logging

### Task 3.9: Create PDF Processor Pipeline
**Time: 3 hours**

Create `services/pdf_processor.py`:
```python
"""PDF processing pipeline coordinator"""
from pathlib import Path
from typing import List, Dict, Any
import logging
from datetime import datetime

from services.ocr_service import OCRService
from services.llm_service import LLMService
from services.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)


class PDFProcessor:
    """Coordinates PDF processing pipeline"""
    
    def __init__(self):
        """Initialize all services"""
        self.ocr = OCRService()
        self.llm = LLMService()
        self.embedding = EmbeddingService()
        logger.info("PDF Processor initialized")
    
    def process_single_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """Process a single PDF file"""
        pdf_path = Path(pdf_path)
        logger.info(f"Processing: {pdf_path.name}")
        
        try:
            # Step 1: OCR
            logger.info("Step 1/3: Running OCR...")
            markdown_text = self.ocr.process_pdf(str(pdf_path))
            
            # Step 2: Extract questions
            logger.info("Step 2/3: Extracting questions...")
            exam_paper = self.llm.extract_questions(
                markdown_text, 
                pdf_path.name
            )
            
            # Step 3: Generate embeddings
            logger.info("Step 3/3: Generating embeddings...")
            question_texts = [
                f"{q.question_text} Topics: {', '.join(q.topics)}"
                for q in exam_paper.questions
            ]
            embeddings = self.embedding.generate_batch_embeddings(question_texts)
            
            # Combine results
            result = {
                'pdf_name': pdf_path.name,
                'year': exam_paper.year,
                'level': exam_paper.level,
                'questions': [],
                'processed_at': datetime.now().isoformat()
            }
            
            for question, embedding in zip(exam_paper.questions, embeddings):
                result['questions'].append({
                    'question_number': question.question_number,
                    'marks': question.marks,
                    'question_text': question.question_text,
                    'topics': question.topics,
                    'question_type': question.question_type,
                    'embedding': embedding
                })
            
            logger.info(f"Successfully processed {pdf_path.name}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to process {pdf_path.name}: {str(e)}")
            raise
    
    def process_folder(self, folder_path: str) -> List[Dict[str, Any]]:
        """Process all PDFs in a folder"""
        folder = Path(folder_path)
        pdf_files = list(folder.glob("*.pdf"))
        
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        results = []
        for i, pdf_file in enumerate(pdf_files):
            logger.info(f"Processing file {i+1}/{len(pdf_files)}")
            try:
                result = self.process_single_pdf(str(pdf_file))
                results.append(result)
            except Exception as e:
                logger.error(f"Skipping {pdf_file.name}: {str(e)}")
                continue
        
        logger.info(f"Completed processing {len(results)} files successfully")
        return results
```

- [ ] Create pipeline coordinator
- [ ] Implement single PDF processing
- [ ] Implement folder processing
- [ ] Add comprehensive logging

### Task 3.10: Test Complete Pipeline
**Time: 1 hour**

Create `test_pipeline.py`:
```python
"""Test complete PDF processing pipeline"""
from services.pdf_processor import PDFProcessor
import json
import logging

logging.basicConfig(level=logging.INFO)

def test_pipeline():
    processor = PDFProcessor()
    
    # Test with a single PDF
    pdf_path = "samples/test.pdf"  # You need to provide this
    
    try:
        result = processor.process_single_pdf(pdf_path)
        
        # Save result to file for inspection
        with open("test_result.json", "w") as f:
            # Remove embeddings for readable output
            for q in result['questions']:
                q['embedding'] = f"[{len(q['embedding'])} dimensions]"
            
            json.dump(result, f, indent=2)
        
        print(f"✅ Successfully processed: {result['pdf_name']}")
        print(f"📊 Found {len(result['questions'])} questions")
        print(f"📅 Year: {result['year']}, Level: {result['level']}")
        print("\nResults saved to test_result.json")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_pipeline()
```

- [ ] Create test script
- [ ] Test with real PDF
- [ ] Verify all components work together

---

## 🌐 Phase 4: Flask API Development (Days 6-7)

### Task 4.1: Create Basic Flask App
**Time: 1 hour**

Create `app.py`:
```python
"""Main Flask application for PDF Question Extractor"""
from flask import Flask, render_template
from flask_cors import CORS
from config import config
import logging
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_app(config_name=None):
    """Create and configure Flask app"""
    app = Flask(__name__)
    
    # Load configuration
    config_name = config_name or os.getenv('FLASK_ENV', 'development')
    app.config.from_object(config[config_name])
    
    # Initialize CORS
    CORS(app)
    
    # Create upload folder
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # Register blueprints
    from api.routes import api_bp
    app.register_blueprint(api_bp, url_prefix='/api')
    
    # Root route
    @app.route('/')
    def index():
        return app.send_static_file('index.html')
    
    logger.info(f"App created with config: {config_name}")
    return app


if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, host='0.0.0.0', port=5000)
```

- [ ] Create Flask app factory
- [ ] Configure CORS
- [ ] Add basic route
- [ ] Test app starts

### Task 4.2: Create Database Session Management
**Time: 1 hour**

Create `database/session.py`:
```python
"""Database session management"""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
from config import Config

# Create engine
engine = create_engine(
    Config.SQLALCHEMY_DATABASE_URI,
    **Config.SQLALCHEMY_ENGINE_OPTIONS
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create scoped session for thread safety
db_session = scoped_session(SessionLocal)


def get_db():
    """Get database session for dependency injection"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Initialize database tables"""
    from database.models import Base
    Base.metadata.create_all(bind=engine)
```

- [ ] Create session management
- [ ] Add dependency injection helper
- [ ] Test database connection

### Task 4.3: Create API Routes Structure
**Time: 2 hours**

Create `api/routes.py`:
```python
"""API routes for PDF Question Extractor"""
from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename
import os
import logging

logger = logging.getLogger(__name__)

# Create blueprint
api_bp = Blueprint('api', __name__)


def allowed_file(filename):
    """Check if file is allowed"""
    from flask import current_app
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in current_app.config['ALLOWED_EXTENSIONS']


@api_bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'message': 'PDF Question Extractor API is running'
    })


@api_bp.route('/upload', methods=['POST'])
def upload_pdfs():
    """Upload PDF files for processing"""
    # TODO: Implement in next task
    pass


@api_bp.route('/questions', methods=['GET'])
def get_questions():
    """Get paginated questions"""
    # TODO: Implement in next task
    pass


@api_bp.route('/questions/<int:question_id>', methods=['PUT'])
def update_question(question_id):
    """Update a single question"""
    # TODO: Implement in next task
    pass


@api_bp.route('/questions/save', methods=['POST'])
def save_approved_questions():
    """Save approved questions to permanent storage"""
    # TODO: Implement in next task
    pass
```

- [ ] Create API blueprint
- [ ] Add health check endpoint
- [ ] Create route stubs
- [ ] Test health endpoint works

### Task 4.4: Implement Upload Endpoint
**Time: 2 hours**

Add to `api/routes.py`:
```python
import uuid
from pathlib import Path
from flask import current_app

# Store processing jobs (in production, use Redis or database)
processing_jobs = {}

@api_bp.route('/upload', methods=['POST'])
def upload_pdfs():
    """Upload PDF files for processing"""
    try:
        if 'pdfs' not in request.files:
            return jsonify({'error': 'No files provided'}), 400
        
        files = request.files.getlist('pdfs')
        if not files:
            return jsonify({'error': 'No files selected'}), 400
        
        # Create job ID
        job_id = str(uuid.uuid4())
        upload_folder = Path(current_app.config['UPLOAD_FOLDER']) / job_id
        upload_folder.mkdir(exist_ok=True)
        
        # Save uploaded files
        saved_files = []
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = upload_folder / filename
                file.save(str(filepath))
                saved_files.append(filename)
                logger.info(f"Saved file: {filename}")
        
        if not saved_files:
            return jsonify({'error': 'No valid PDF files'}), 400
        
        # Store job info
        processing_jobs[job_id] = {
            'status': 'uploaded',
            'files': saved_files,
            'total': len(saved_files),
            'processed': 0,
            'folder': str(upload_folder)
        }
        
        # TODO: Trigger processing (will add in later task)
        
        return jsonify({
            'job_id': job_id,
            'files_uploaded': len(saved_files),
            'message': 'Files uploaded successfully'
        }), 200
        
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        return jsonify({'error': str(e)}), 500
```

- [ ] Implement file upload
- [ ] Add file validation
- [ ] Create job tracking
- [ ] Test with Postman/curl

### Task 4.5: Implement Questions Endpoints
**Time: 3 hours**

Add to `api/routes.py`:
```python
from database.session import get_db
from database.models import ExtractedQuestion
from sqlalchemy import func

@api_bp.route('/questions', methods=['GET'])
def get_questions():
    """Get paginated questions"""
    try:
        # Get query parameters
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 20, type=int)
        status = request.args.get('status', None)
        source_pdf = request.args.get('source', None)
        
        # Get database session
        db = next(get_db())
        
        # Build query
        query = db.query(ExtractedQuestion)
        
        # Apply filters
        if status:
            query = query.filter(ExtractedQuestion.status == status)
        if source_pdf:
            query = query.filter(ExtractedQuestion.source_pdf == source_pdf)
        
        # Get total count
        total = query.count()
        
        # Apply pagination
        questions = query.offset((page - 1) * per_page).limit(per_page).all()
        
        # Convert to dict
        questions_data = [q.to_dict() for q in questions]
        
        return jsonify({
            'questions': questions_data,
            'total': total,
            'page': page,
            'per_page': per_page,
            'total_pages': (total + per_page - 1) // per_page
        }), 200
        
    except Exception as e:
        logger.error(f"Get questions error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@api_bp.route('/questions/<int:question_id>', methods=['PUT'])
def update_question(question_id):
    """Update a single question"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        db = next(get_db())
        
        # Find question
        question = db.query(ExtractedQuestion).filter_by(id=question_id).first()
        if not question:
            return jsonify({'error': 'Question not found'}), 404
        
        # Update fields
        updateable_fields = [
            'question_number', 'marks', 'year', 'level', 
            'topics', 'question_type', 'question_text', 'status'
        ]
        
        for field in updateable_fields:
            if field in data:
                setattr(question, field, data[field])
        
        # Mark as modified
        question.modified = True
        
        # Save changes
        db.commit()
        
        return jsonify({
            'message': 'Question updated successfully',
            'question': question.to_dict()
        }), 200
        
    except Exception as e:
        logger.error(f"Update question error: {str(e)}")
        return jsonify({'error': str(e)}), 500
```

- [ ] Implement pagination logic
- [ ] Add filtering options
- [ ] Implement update endpoint
- [ ] Test all endpoints

### Task 4.6: Implement Save Approved Questions
**Time: 2 hours**

Add to `api/routes.py`:
```python
from database.models import Question, QuestionEmbedding
from services.embedding_service import EmbeddingService

@api_bp.route('/questions/save', methods=['POST'])
def save_approved_questions():
    """Save approved questions to permanent storage"""
    try:
        data = request.get_json()
        if not data or 'question_ids' not in data:
            return jsonify({'error': 'No question IDs provided'}), 400
        
        db = next(get_db())
        embedding_service = EmbeddingService()
        
        # Get approved questions
        approved_questions = db.query(ExtractedQuestion).filter(
            ExtractedQuestion.id.in_(data['question_ids']),
            ExtractedQuestion.status == 'approved'
        ).all()
        
        if not approved_questions:
            return jsonify({'error': 'No approved questions found'}), 404
        
        saved_count = 0
        
        for extracted_q in approved_questions:
            try:
                # Create permanent question
                question = Question.from_extracted(extracted_q)
                db.add(question)
                db.flush()  # Get the ID
                
                # Generate embedding
                embedding_text = f"{question.question_text} Topics: {', '.join(question.topics or [])}"
                embedding = embedding_service.generate_embedding(embedding_text)
                
                # Save embedding
                question_embedding = QuestionEmbedding(
                    question_id=question.id,
                    embedding=embedding,
                    model_name='gemini-embedding-001',
                    model_version='1.0'
                )
                db.add(question_embedding)
                
                # Mark extracted question as processed
                extracted_q.status = 'processed'
                
                saved_count += 1
                
            except Exception as e:
                logger.error(f"Failed to save question {extracted_q.id}: {str(e)}")
                continue
        
        # Commit all changes
        db.commit()
        
        return jsonify({
            'message': f'Saved {saved_count} questions successfully',
            'saved_count': saved_count
        }), 200
        
    except Exception as e:
        logger.error(f"Save questions error: {str(e)}")
        return jsonify({'error': str(e)}), 500
```

- [ ] Implement save logic
- [ ] Add embedding generation
- [ ] Update question status
- [ ] Test save functionality

---

## 🎨 Phase 5: Frontend Development (Days 8-9)

### Task 5.1: Create HTML Structure
**Time: 1 hour**

Update `static/index.html`:
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PDF Question Extractor</title>
    
    <!-- Tabulator CSS -->
    <link href="https://unpkg.com/tabulator-tables@5.5/dist/css/tabulator.min.css" rel="stylesheet">
    
    <!-- Custom CSS -->
    <link rel="stylesheet" href="/static/css/style.css">
</head>
<body>
    <div class="container">
        <!-- Header -->
        <header>
            <h1>📄 PDF Question Extractor</h1>
            <p>Extract and review questions from exam papers</p>
        </header>
        
        <!-- Upload Section -->
        <section class="upload-section">
            <h2>Upload PDFs</h2>
            <div class="upload-area">
                <input type="file" id="pdfFiles" multiple accept=".pdf" style="display: none;">
                <button id="selectFiles" class="btn btn-primary">
                    Select PDF Files
                </button>
                <div id="fileList"></div>
                <button id="uploadFiles" class="btn btn-success" style="display: none;">
                    Upload and Process
                </button>
            </div>
            <div id="uploadProgress" class="progress" style="display: none;">
                <div class="progress-bar"></div>
                <span class="progress-text">Uploading...</span>
            </div>
        </section>
        
        <!-- Review Section -->
        <section class="review-section" style="display: none;">
            <h2>Review Questions</h2>
            
            <!-- Controls -->
            <div class="controls">
                <button id="approveAll" class="btn btn-success">
                    ✓ Approve All
                </button>
                <button id="rejectAll" class="btn btn-danger">
                    ✗ Reject All
                </button>
                <button id="saveApproved" class="btn btn-primary">
                    💾 Save Approved
                </button>
                <button id="exportData" class="btn btn-secondary">
                    📥 Export CSV
                </button>
            </div>
            
            <!-- Filters -->
            <div class="filters">
                <select id="statusFilter">
                    <option value="">All Status</option>
                    <option value="pending">Pending</option>
                    <option value="approved">Approved</option>
                    <option value="rejected">Rejected</option>
                </select>
                <select id="sourceFilter">
                    <option value="">All Sources</option>
                </select>
            </div>
            
            <!-- Table -->
            <div id="questionTable"></div>
            
            <!-- Status Bar -->
            <div class="status-bar">
                <span id="statusText">0 questions loaded</span>
                <span id="saveStatus"></span>
            </div>
        </section>
    </div>
    
    <!-- Scripts -->
    <script src="https://unpkg.com/tabulator-tables@5.5/dist/js/tabulator.min.js"></script>
    <script src="/static/js/app.js"></script>
</body>
</html>
```

- [ ] Create HTML structure
- [ ] Add all sections
- [ ] Include scripts
- [ ] Test page loads

### Task 5.2: Style the Application
**Time: 1 hour**

Create `static/css/style.css`:
```css
/* Reset and base styles */
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background-color: #f5f5f5;
    color: #333;
    line-height: 1.6;
}

.container {
    max-width: 1400px;
    margin: 0 auto;
    padding: 20px;
}

/* Header styles */
header {
    text-align: center;
    margin-bottom: 40px;
    padding: 20px;
    background: white;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

header h1 {
    color: #2c3e50;
    margin-bottom: 10px;
}

header p {
    color: #7f8c8d;
}

/* Section styles */
section {
    background: white;
    padding: 30px;
    margin-bottom: 30px;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

section h2 {
    color: #2c3e50;
    margin-bottom: 20px;
    padding-bottom: 10px;
    border-bottom: 2px solid #ecf0f1;
}

/* Upload area styles */
.upload-area {
    border: 2px dashed #bdc3c7;
    border-radius: 8px;
    padding: 40px;
    text-align: center;
    background-color: #ecf0f1;
    margin-bottom: 20px;
}

#fileList {
    margin: 20px 0;
    text-align: left;
}

.file-item {
    padding: 8px;
    margin: 5px 0;
    background: white;
    border-radius: 4px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

/* Button styles */
.btn {
    padding: 10px 20px;
    border: none;
    border-radius: 4px;
    cursor: pointer;
    font-size: 16px;
    transition: all 0.3s ease;
    margin-right: 10px;
}

.btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(0,0,0,0.2);
}

.btn-primary {
    background-color: #3498db;
    color: white;
}

.btn-primary:hover {
    background-color: #2980b9;
}

.btn-success {
    background-color: #27ae60;
    color: white;
}

.btn-success:hover {
    background-color: #229954;
}

.btn-danger {
    background-color: #e74c3c;
    color: white;
}

.btn-danger:hover {
    background-color: #c0392b;
}

.btn-secondary {
    background-color: #95a5a6;
    color: white;
}

.btn-secondary:hover {
    background-color: #7f8c8d;
}

/* Progress bar styles */
.progress {
    height: 30px;
    background-color: #ecf0f1;
    border-radius: 4px;
    overflow: hidden;
    position: relative;
}

.progress-bar {
    height: 100%;
    background-color: #3498db;
    width: 0%;
    transition: width 0.3s ease;
}

.progress-text {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    color: #333;
    font-weight: bold;
}

/* Controls and filters */
.controls, .filters {
    margin-bottom: 20px;
}

.filters select {
    padding: 8px 12px;
    margin-right: 10px;
    border: 1px solid #bdc3c7;
    border-radius: 4px;
    font-size: 14px;
}

/* Table styles */
#questionTable {
    margin-bottom: 20px;
}

/* Modified row indicator */
.tabulator-row.modified {
    background-color: #fff3cd !important;
}

/* Status bar */
.status-bar {
    display: flex;
    justify-content: space-between;
    padding: 10px;
    background-color: #ecf0f1;
    border-radius: 4px;
    font-size: 14px;
}

#saveStatus {
    color: #27ae60;
    font-weight: bold;
}

/* Responsive design */
@media (max-width: 768px) {
    .container {
        padding: 10px;
    }
    
    section {
        padding: 20px;
    }
    
    .btn {
        display: block;
        width: 100%;
        margin-bottom: 10px;
    }
    
    .controls {
        display: flex;
        flex-direction: column;
    }
}
```

- [ ] Add all styles
- [ ] Make responsive
- [ ] Test appearance

### Task 5.3: Implement File Upload JavaScript
**Time: 2 hours**

Start `static/js/app.js`:
```javascript
// Global variables
let selectedFiles = [];
let table = null;
let currentJobId = null;

// DOM elements
const pdfFilesInput = document.getElementById('pdfFiles');
const selectFilesBtn = document.getElementById('selectFiles');
const fileListDiv = document.getElementById('fileList');
const uploadFilesBtn = document.getElementById('uploadFiles');
const uploadProgress = document.getElementById('uploadProgress');
const progressBar = document.querySelector('.progress-bar');
const progressText = document.querySelector('.progress-text');
const reviewSection = document.querySelector('.review-section');

// File selection
selectFilesBtn.addEventListener('click', () => {
    pdfFilesInput.click();
});

pdfFilesInput.addEventListener('change', (e) => {
    handleFileSelection(e.target.files);
});

function handleFileSelection(files) {
    selectedFiles = Array.from(files).filter(file => 
        file.type === 'application/pdf'
    );
    
    displaySelectedFiles();
    
    if (selectedFiles.length > 0) {
        uploadFilesBtn.style.display = 'block';
    } else {
        uploadFilesBtn.style.display = 'none';
        alert('Please select PDF files only');
    }
}

function displaySelectedFiles() {
    fileListDiv.innerHTML = '';
    
    selectedFiles.forEach((file, index) => {
        const fileItem = document.createElement('div');
        fileItem.className = 'file-item';
        fileItem.innerHTML = `
            <span>📄 ${file.name} (${formatFileSize(file.size)})</span>
            <button onclick="removeFile(${index})" class="btn btn-sm btn-danger">
                Remove
            </button>
        `;
        fileListDiv.appendChild(fileItem);
    });
}

function removeFile(index) {
    selectedFiles.splice(index, 1);
    displaySelectedFiles();
    
    if (selectedFiles.length === 0) {
        uploadFilesBtn.style.display = 'none';
    }
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

// File upload
uploadFilesBtn.addEventListener('click', uploadFiles);

async function uploadFiles() {
    if (selectedFiles.length === 0) return;
    
    const formData = new FormData();
    selectedFiles.forEach(file => {
        formData.append('pdfs', file);
    });
    
    // Show progress
    uploadProgress.style.display = 'block';
    uploadFilesBtn.disabled = true;
    
    try {
        const response = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`Upload failed: ${response.statusText}`);
        }
        
        const result = await response.json();
        currentJobId = result.job_id;
        
        // Update progress
        progressBar.style.width = '100%';
        progressText.textContent = 'Upload complete! Processing...';
        
        // Clear file selection
        selectedFiles = [];
        fileListDiv.innerHTML = '';
        uploadFilesBtn.style.display = 'none';
        pdfFilesInput.value = '';
        
        // Start processing (mock for now)
        setTimeout(() => {
            uploadProgress.style.display = 'none';
            reviewSection.style.display = 'block';
            loadQuestions();
        }, 2000);
        
    } catch (error) {
        console.error('Upload error:', error);
        alert(`Upload failed: ${error.message}`);
        
        uploadProgress.style.display = 'none';
        uploadFilesBtn.disabled = false;
        progressBar.style.width = '0%';
    }
}

// Make removeFile function global
window.removeFile = removeFile;
```

- [ ] Implement file selection
- [ ] Add file display
- [ ] Implement upload
- [ ] Add progress tracking

### Task 5.4: Implement Tabulator Table
**Time: 3 hours**

Add to `static/js/app.js`:
```javascript
// Initialize Tabulator
function initTable() {
    table = new Tabulator("#questionTable", {
        height: "600px",
        layout: "fitColumns",
        pagination: "local",
        paginationSize: 20,
        movableColumns: true,
        columns: [
            {
                title: "✓", 
                field: "approved", 
                formatter: "tickCross",
                editor: true,
                width: 50,
                hozAlign: "center",
                cellClick: function(e, cell) {
                    const row = cell.getRow();
                    const data = row.getData();
                    
                    // Update status based on checkbox
                    if (data.approved) {
                        data.status = 'approved';
                    } else {
                        data.status = 'rejected';
                    }
                    
                    row.update(data);
                    markModified(row);
                }
            },
            {
                title: "Q#",
                field: "question_number",
                editor: "input",
                width: 80
            },
            {
                title: "Question",
                field: "question_text",
                editor: "textarea",
                formatter: "textarea",
                minWidth: 300
            },
            {
                title: "Marks",
                field: "marks",
                editor: "number",
                width: 80
            },
            {
                title: "Topics",
                field: "topics",
                editor: "input",
                formatter: function(cell) {
                    const topics = cell.getValue();
                    return Array.isArray(topics) ? topics.join(", ") : "";
                },
                width: 200
            },
            {
                title: "Type",
                field: "question_type",
                editor: "select",
                editorParams: {
                    values: ["MCQ", "Essay", "Short Answer", "True/False", "Fill in the Blank"]
                },
                width: 120
            },
            {
                title: "Year",
                field: "year",
                editor: "input",
                width: 80
            },
            {
                title: "Level",
                field: "level",
                editor: "input",
                width: 100
            },
            {
                title: "Source",
                field: "source_pdf",
                width: 150
            },
            {
                title: "Status",
                field: "status",
                width: 100,
                formatter: function(cell) {
                    const status = cell.getValue();
                    const colors = {
                        'pending': '#f39c12',
                        'approved': '#27ae60',
                        'rejected': '#e74c3c'
                    };
                    return `<span style="color: ${colors[status] || '#333'}">${status}</span>`;
                }
            }
        ],
        cellEdited: function(cell) {
            markModified(cell.getRow());
            scheduleAutoSave();
        }
    });
}

// Mark row as modified
function markModified(row) {
    row.getElement().classList.add("modified");
    const data = row.getData();
    data.modified = true;
    row.update(data);
}

// Auto-save functionality
let autoSaveTimer;

function scheduleAutoSave() {
    clearTimeout(autoSaveTimer);
    autoSaveTimer = setTimeout(autoSave, 1000);
}

async function autoSave() {
    const modifiedData = table.getData().filter(q => q.modified);
    
    if (modifiedData.length === 0) return;
    
    for (const question of modifiedData) {
        try {
            await updateQuestion(question);
            question.modified = false;
        } catch (error) {
            console.error(`Failed to save question ${question.id}:`, error);
        }
    }
    
    document.getElementById('saveStatus').textContent = `Auto-saved ${modifiedData.length} changes`;
    setTimeout(() => {
        document.getElementById('saveStatus').textContent = '';
    }, 3000);
}

async function updateQuestion(questionData) {
    const response = await fetch(`/api/questions/${questionData.id}`, {
        method: 'PUT',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(questionData)
    });
    
    if (!response.ok) {
        throw new Error('Failed to update question');
    }
}
```

- [ ] Initialize Tabulator
- [ ] Configure columns
- [ ] Add cell editing
- [ ] Implement auto-save

### Task 5.5: Implement Question Operations
**Time: 2 hours**

Add to `static/js/app.js`:
```javascript
// Load questions from API
async function loadQuestions(page = 1) {
    try {
        const response = await fetch(`/api/questions?page=${page}&per_page=20`);
        
        if (!response.ok) {
            throw new Error('Failed to load questions');
        }
        
        const data = await response.json();
        
        // Initialize table if not exists
        if (!table) {
            initTable();
        }
        
        // Transform data for display
        const questions = data.questions.map(q => ({
            ...q,
            approved: q.status === 'approved'
        }));
        
        // Set data
        table.setData(questions);
        
        // Update status
        document.getElementById('statusText').textContent = 
            `${data.total} questions loaded (Page ${data.page} of ${data.total_pages})`;
        
        // Update source filter
        updateSourceFilter(questions);
        
    } catch (error) {
        console.error('Load questions error:', error);
        alert('Failed to load questions');
    }
}

// Update source filter dropdown
function updateSourceFilter(questions) {
    const sourceFilter = document.getElementById('sourceFilter');
    const sources = [...new Set(questions.map(q => q.source_pdf))];
    
    sourceFilter.innerHTML = '<option value="">All Sources</option>';
    sources.forEach(source => {
        const option = document.createElement('option');
        option.value = source;
        option.textContent = source;
        sourceFilter.appendChild(option);
    });
}

// Bulk operations
document.getElementById('approveAll').addEventListener('click', () => {
    table.getData().forEach(row => {
        table.updateData([{
            id: row.id,
            approved: true,
            status: 'approved'
        }]);
    });
    
    // Mark all as modified for saving
    table.getRows().forEach(row => markModified(row));
    scheduleAutoSave();
});

document.getElementById('rejectAll').addEventListener('click', () => {
    table.getData().forEach(row => {
        table.updateData([{
            id: row.id,
            approved: false,
            status: 'rejected'
        }]);
    });
    
    // Mark all as modified for saving
    table.getRows().forEach(row => markModified(row));
    scheduleAutoSave();
});

// Save approved questions
document.getElementById('saveApproved').addEventListener('click', async () => {
    const approvedQuestions = table.getData().filter(q => q.status === 'approved');
    
    if (approvedQuestions.length === 0) {
        alert('No approved questions to save');
        return;
    }
    
    if (!confirm(`Save ${approvedQuestions.length} approved questions to permanent storage?`)) {
        return;
    }
    
    try {
        const response = await fetch('/api/questions/save', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                question_ids: approvedQuestions.map(q => q.id)
            })
        });
        
        if (!response.ok) {
            throw new Error('Failed to save questions');
        }
        
        const result = await response.json();
        alert(result.message);
        
        // Reload questions
        loadQuestions();
        
    } catch (error) {
        console.error('Save error:', error);
        alert('Failed to save questions');
    }
});

// Export functionality
document.getElementById('exportData').addEventListener('click', () => {
    const data = table.getData();
    
    // Convert to CSV
    const csv = convertToCSV(data);
    
    // Download
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `questions_export_${new Date().toISOString().slice(0, 10)}.csv`;
    a.click();
});

function convertToCSV(data) {
    const headers = [
        'Question Number', 'Question Text', 'Marks', 'Topics',
        'Type', 'Year', 'Level', 'Source PDF', 'Status'
    ];
    
    const rows = data.map(q => [
        q.question_number || '',
        `"${(q.question_text || '').replace(/"/g, '""')}"`,
        q.marks || '',
        `"${(q.topics || []).join(', ')}"`,
        q.question_type || '',
        q.year || '',
        q.level || '',
        q.source_pdf || '',
        q.status || ''
    ]);
    
    return [headers.join(','), ...rows.map(r => r.join(','))].join('\n');
}

// Filters
document.getElementById('statusFilter').addEventListener('change', (e) => {
    if (e.target.value) {
        table.setFilter("status", "=", e.target.value);
    } else {
        table.clearFilter();
    }
});

document.getElementById('sourceFilter').addEventListener('change', (e) => {
    if (e.target.value) {
        table.setFilter("source_pdf", "=", e.target.value);
    } else {
        table.clearFilter();
    }
});

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    // Check if we should load questions immediately (for testing)
    const urlParams = new URLSearchParams(window.location.search);
    if (urlParams.get('test') === 'true') {
        reviewSection.style.display = 'block';
        loadQuestions();
    }
});
```

- [ ] Implement question loading
- [ ] Add bulk operations
- [ ] Implement save functionality
- [ ] Add export feature
- [ ] Test all operations

---

## 🧪 Phase 6: Testing & Integration (Days 10-11)

### Task 6.1: Create Test Data Generator
**Time: 1 hour**

Create `tests/generate_test_data.py`:
```python
"""Generate test data for development"""
from database.session import get_db
from database.models import ExtractedQuestion
import random
from datetime import datetime

def generate_test_questions(count=50):
    """Generate sample questions for testing"""
    db = next(get_db())
    
    question_types = ["MCQ", "Essay", "Short Answer", "True/False"]
    topics = ["Mathematics", "Physics", "Chemistry", "Biology", "History", "Geography"]
    years = ["2021", "2022", "2023", "2024"]
    levels = ["Grade 10", "Grade 11", "Grade 12", "University"]
    
    for i in range(count):
        question = ExtractedQuestion(
            question_number=f"{i+1}",
            marks=random.randint(1, 20),
            year=random.choice(years),
            level=random.choice(levels),
            topics=random.sample(topics, k=random.randint(1, 3)),
            question_type=random.choice(question_types),
            question_text=f"This is sample question {i+1}. What is the answer to this question?",
            source_pdf=f"sample_exam_{random.randint(1, 5)}.pdf",
            status="pending"
        )
        db.add(question)
    
    db.commit()
    print(f"Generated {count} test questions")

if __name__ == "__main__":
    generate_test_questions()
```

- [ ] Create test data script
- [ ] Generate sample questions
- [ ] Test UI with data

### Task 6.2: Create Integration Tests
**Time: 2 hours**

Create `tests/test_integration.py`:
```python
"""Integration tests for PDF Question Extractor"""
import unittest
import json
from app import create_app
from database.session import engine
from database.models import Base

class IntegrationTests(unittest.TestCase):
    
    def setUp(self):
        """Set up test client"""
        self.app = create_app('testing')
        self.client = self.app.test_client()
        
        # Create tables
        Base.metadata.create_all(bind=engine)
    
    def tearDown(self):
        """Clean up after tests"""
        Base.metadata.drop_all(bind=engine)
    
    def test_health_check(self):
        """Test health endpoint"""
        response = self.client.get('/api/health')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'healthy')
    
    def test_questions_endpoint(self):
        """Test questions listing"""
        response = self.client.get('/api/questions')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('questions', data)
        self.assertIn('total', data)
    
    def test_file_upload(self):
        """Test PDF upload"""
        # Create a dummy PDF file
        with open('test.pdf', 'rb') as f:
            response = self.client.post(
                '/api/upload',
                data={'pdfs': (f, 'test.pdf')},
                content_type='multipart/form-data'
            )
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('job_id', data)

if __name__ == '__main__':
    unittest.main()
```

- [ ] Create test file
- [ ] Test all endpoints
- [ ] Verify functionality

### Task 6.3: Create End-to-End Test Script
**Time: 1 hour**

Create `tests/test_e2e.py`:
```python
"""End-to-end test of complete pipeline"""
import asyncio
from pathlib import Path
from services.pdf_processor import PDFProcessor
from database.session import get_db
from database.models import ExtractedQuestion

async def test_full_pipeline():
    """Test complete processing pipeline"""
    print("Starting end-to-end test...")
    
    # Initialize processor
    processor = PDFProcessor()
    
    # Process test PDF
    test_pdf = Path("samples/test.pdf")
    if not test_pdf.exists():
        print("❌ Please add a test PDF to samples/test.pdf")
        return
    
    try:
        # Process PDF
        print("📄 Processing PDF...")
        result = await processor.process_single_pdf(str(test_pdf))
        
        print(f"✅ Found {len(result['questions'])} questions")
        
        # Save to database
        print("💾 Saving to database...")
        db = next(get_db())
        
        for q_data in result['questions']:
            question = ExtractedQuestion(
                question_number=q_data['question_number'],
                marks=q_data['marks'],
                year=result['year'],
                level=result['level'],
                topics=q_data['topics'],
                question_type=q_data['question_type'],
                question_text=q_data['question_text'],
                source_pdf=result['pdf_name']
            )
            db.add(question)
        
        db.commit()
        print("✅ Saved to database successfully")
        
        # Verify
        count = db.query(ExtractedQuestion).count()
        print(f"📊 Total questions in database: {count}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_full_pipeline())
```

- [ ] Create E2E test
- [ ] Test full pipeline
- [ ] Verify all components

---

## 🚀 Phase 7: Final Setup & Deployment (Day 12)

### Task 7.1: Create Startup Script
**Time: 30 minutes**

Create `run.sh`:
```bash
#!/bin/bash

echo "🚀 Starting PDF Question Extractor..."

# Activate virtual environment
source venv/bin/activate

# Check database
echo "🔍 Checking database..."
python -c "from database.init_db import verify_database; verify_database()"

# Start Flask app
echo "🌐 Starting Flask server..."
python app.py
```

Make it executable:
```bash
chmod +x run.sh
```

- [ ] Create startup script
- [ ] Make executable
- [ ] Test script works

### Task 7.2: Create Setup Instructions
**Time: 1 hour**

Create `SETUP.md`:
```markdown
# PDF Question Extractor - Setup Guide

## Quick Setup

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure environment**:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and database credentials
   ```

3. **Initialize database**:
   ```bash
   python database/init_db.py
   ```

4. **Run the application**:
   ```bash
   python app.py
   # Or use the startup script:
   ./run.sh
   ```

5. **Access the application**:
   Open http://localhost:5000 in your browser

## Testing

1. **Generate test data**:
   ```bash
   python tests/generate_test_data.py
   ```

2. **Run tests**:
   ```bash
   python -m pytest tests/
   ```

## Troubleshooting

### Database connection issues
- Check PostgreSQL is running
- Verify credentials in .env
- Ensure pgvector extension is installed

### API key issues
- Verify keys are correct in .env
- Check API quotas/limits
- Ensure internet connection

### Processing errors
- Check PDF file size (<50MB)
- Verify PDF is not corrupted
- Check logs for specific errors
```

- [ ] Create setup guide
- [ ] Add troubleshooting
- [ ] Test instructions work

### Task 7.3: Final Testing Checklist
**Time: 2 hours**

Run through complete testing:

- [ ] **Environment Setup**
  - [ ] Virtual environment works
  - [ ] All packages installed
  - [ ] Environment variables loaded

- [ ] **Database**
  - [ ] PostgreSQL connects
  - [ ] Tables created
  - [ ] Indexes working
  - [ ] pgvector queries work

- [ ] **API Services**
  - [ ] OCR service processes PDFs
  - [ ] LLM extracts questions
  - [ ] Embeddings generate correctly
  - [ ] Pipeline processes files

- [ ] **Web Application**
  - [ ] Homepage loads
  - [ ] File upload works
  - [ ] Questions display in table
  - [ ] Editing saves changes
  - [ ] Bulk operations work
  - [ ] Export functionality works

- [ ] **End-to-End**
  - [ ] Upload PDF
  - [ ] View extracted questions
  - [ ] Edit and approve questions
  - [ ] Save to permanent storage
  - [ ] Verify in database

### Task 7.4: Create Documentation
**Time: 1 hour**

Update README with:

- [ ] Project overview
- [ ] Features list
- [ ] Installation guide
- [ ] Usage instructions
- [ ] API documentation
- [ ] Troubleshooting section
- [ ] Contributing guidelines

---

## 🎉 Completion Checklist

### Core Functionality
- [ ] Project structure created
- [ ] Database configured with pgvector
- [ ] OCR service working
- [ ] LLM extraction working
- [ ] Embedding generation working
- [ ] Flask API functional
- [ ] Web UI complete
- [ ] All features tested

### Documentation
- [ ] README.md complete
- [ ] SETUP.md created
- [ ] Code commented
- [ ] API documented

### Quality
- [ ] Error handling implemented
- [ ] Logging configured
- [ ] Tests written
- [ ] Performance acceptable

### Deployment Ready
- [ ] Environment variables documented
- [ ] Dependencies listed
- [ ] Startup scripts created
- [ ] Instructions clear

## 🏆 Congratulations!

You've built a complete PDF Question Extractor application! The system can:
- Process PDF files with OCR
- Extract questions using AI
- Store questions with vector embeddings
- Provide a web interface for review
- Enable semantic search capabilities

## Next Steps

1. **Enhance Features**:
   - Add user authentication
   - Implement duplicate detection
   - Add more export formats
   - Create admin dashboard

2. **Optimize Performance**:
   - Add Redis for job queuing
   - Implement parallel processing
   - Cache OCR results
   - Optimize database queries

3. **Improve UI**:
   - Add dark mode
   - Enhance mobile experience
   - Add keyboard shortcuts
   - Implement drag-and-drop

Remember: Take breaks, test frequently, and don't hesitate to ask for help!