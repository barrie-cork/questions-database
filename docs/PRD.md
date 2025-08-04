**Product Requirements Document (PRD)**

**Product Name:** Local PDF Question Extractor

**Prepared for:** \[User]
**Prepared by:** PRD Writer GPT
**Date:** 2025-08-04

---

### 1. **Overview**

A local application for extracting questions from scanned and digital PDF test files using Mistral OCR API's advanced document understanding capabilities and Gemini 2.5 Flash Lite for intelligent question extraction, with an interface for manual approval and final storage into a PostgreSQL database with vector search capabilities.

---

### 2. **Goals and Objectives**

* Extract full questions from mixed-format PDFs (scanned or digital)
* Include metadata such as question number, marks, year, level, topic, type, and source filename
* Allow user review before saving
* Store approved questions into a PostgreSQL database with vector embeddings for semantic search

---

### 3. **User Stories**

1. *As a user*, I want to upload a folder of PDFs so that the system can process them all at once.
2. *As a user*, I want questions to be extracted automatically using best-effort logic so I don’t have to manually parse them.
3. *As a user*, I want to approve or reject extracted questions via a simple web interface.
4. *As a user*, I want approved questions to be saved into a SQL database with full metadata.

---

### 4. **Core Features**

#### 4.1 Input Handling

* Folder-based PDF ingestion
* Local app launched from VS Code

#### 4.2 Preprocessing

* Use **Mistral OCR API** (mistral-ocr-latest) to process PDFs:
  * Accepts PDF and image files (max 50MB, 1000 pages)
  * Extracts text, tables, equations, and complex layouts
  * Preserves document structure (headers, lists, formatting)
  * Outputs structured Markdown with interleaved text and images
  * Handles multilingual content and various scripts
  * Processes up to 2,000 pages/minute
* Extract image bounding boxes and metadata for reference
* Utilize ~94.9% accuracy for reliable text extraction

#### 4.3 Question Extraction & Embedding

* Use **Gemini 2.5 Flash** for intelligent question parsing:
  * 1M token context window (handles entire exam papers)
  * Native JSON schema support for structured output using google-genai SDK
  * Cost-effective with 22% efficiency gains vs previous models
  * Lower latency than other models
  * Best price/performance ratio for extraction tasks
* Smart chunking strategy using LangChain:
  * Process whole documents <50k characters in one pass
  * Recursive text splitting for larger documents
  * Maintains context with 200-character overlap
* Extract the following data with guaranteed structure:
  * Question number
  * Marks
  * Year of the paper
  * Level of the paper
  * Topics/Sub-topics (as array)
  * Question type (MCQ, Essay, Short Answer, etc.)
  * Full question text
  * Source PDF filename
* Generate **Gemini embeddings** for each question:
  * Use `gemini-embedding-001` model (3072 dimensions, configurable)
  * Support for dimension reduction (768/1536/3072) with Matryoshka learning
  * Supports up to 8K tokens per embedding
  * Enhanced embeddings with cognitive skill inference
  * Store embeddings separately for versioning

#### 4.4 Review Interface (Web UI)

* **Tabulator.js** data grid for question review:
  * Paginated view (20 questions per page)
  * Inline editing with validation
  * Keyboard navigation support
  * Auto-save on cell edit
* Features:
  * Approve/reject checkbox per question
  * Bulk operations (approve/reject all)
  * Filter by status, PDF source, question type
  * Export to CSV functionality
  * Visual status indicators (approved/rejected/modified)
* Final submission saves only approved questions to permanent database

#### 4.5 Storage

* Two-stage storage system:
  * **Temporary storage** for extracted questions (JSON file or temp DB table)
  * **Permanent storage** for approved questions only
* Database: **PostgreSQL** with **pgvector** extension for vector similarity search
* Schema:

  ```sql
  -- Enable required extensions
  CREATE EXTENSION IF NOT EXISTS vector;
  CREATE EXTENSION IF NOT EXISTS pg_trgm;  -- For text search
  
  -- Temporary storage for review
  CREATE TABLE extracted_questions (
    id BIGSERIAL PRIMARY KEY,
    question_number TEXT,
    marks INTEGER,
    year TEXT,
    level TEXT,
    topics TEXT[],  -- Native PostgreSQL array
    question_type TEXT,
    question_text TEXT NOT NULL,
    source_pdf TEXT NOT NULL,
    status TEXT DEFAULT 'pending',  -- pending/approved/rejected
    modified BOOLEAN DEFAULT FALSE,
    extraction_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB  -- Flexible additional data
  );

  -- Main questions table
  CREATE TABLE questions (
    id BIGSERIAL PRIMARY KEY,
    question_number TEXT,
    marks INTEGER,
    year TEXT,
    level TEXT,
    topics TEXT[],  -- Native PostgreSQL array
    question_type TEXT,
    question_text TEXT NOT NULL,
    source_pdf TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB  -- Additional flexible data
  );
  
  -- Question embeddings table (separate for versioning)
  CREATE TABLE question_embeddings (
    id BIGSERIAL PRIMARY KEY,
    question_id BIGINT NOT NULL,
    embedding vector(768),  -- Gemini embedding (768 dimensions)
    model_name VARCHAR(100) NOT NULL DEFAULT 'gemini-embedding-001',
    model_version VARCHAR(50) NOT NULL,
    embedding_generated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_question FOREIGN KEY (question_id) 
        REFERENCES questions(id) ON DELETE CASCADE,
    CONSTRAINT unique_question_model 
        UNIQUE (question_id, model_name, model_version)
  );
  
  -- Create indexes for performance
  CREATE INDEX idx_questions_source ON questions(source_pdf);
  CREATE INDEX idx_questions_year ON questions(year);
  CREATE INDEX idx_questions_type ON questions(question_type);
  CREATE INDEX idx_questions_topics ON questions USING GIN(topics);
  CREATE INDEX idx_questions_metadata ON questions USING GIN(metadata);
  CREATE INDEX idx_questions_text_search 
    ON questions USING GIN(to_tsvector('english', question_number || ' ' || question_text));
  
  -- Create HNSW index for vector similarity search
  CREATE INDEX idx_question_embeddings_hnsw 
    ON question_embeddings 
    USING hnsw (embedding vector_cosine_ops);
  ```

---

### 5. **Tech Stack**

* **OCR**: Mistral OCR API (mistral-ocr-latest)
  * Pricing: $1 per 1,000 pages ($0.001/page)
  * Available via la Plateforme API
  * Optional self-hosting for data privacy
* **LLM Parsing**: Gemini 2.5 Flash
  * Best price/performance ratio with 22% efficiency gains
  * Native structured output with JSON schema using google-genai SDK
  * 1M token context window with thinking mode
  * Model: `gemini-2.5-flash` (stable as of 2025)
* **Backend**: FastAPI 0.115+
  * Modern async REST API
  * Automatic OpenAPI documentation
  * Built-in request validation
  * WebSocket support for real-time updates
* **Frontend**: HTML5 + CSS3 + Vanilla JS + Tabulator.js
  * Tabulator.js 5.5 for data grid
  * No build process required
  * CDN-hosted dependencies
* **Python Libraries**:
  * `mistralai==1.2.3` - Mistral OCR client
  * `google-genai==0.9.0` - New Gemini API SDK (replaces deprecated google-generativeai)
  * `langchain-text-splitters==0.3.4` - Smart text chunking
  * `pydantic==2.10.4` - Schema validation
  * `fastapi==0.115.0` - Modern async web framework
  * `uvicorn[standard]==0.34.0` - ASGI server
  * `sqlalchemy[asyncio]==2.0.35` - Database ORM with async support
  * `asyncpg==0.30.0` - Async PostgreSQL driver
  * `tenacity==9.0.0` - Retry logic with exponential backoff
  * `httpx==0.28.0` - Async HTTP client
  * `pgvector==0.3.6` - PostgreSQL vector extension client
  * `aiofiles==24.1.0` - Async file operations
* **Database**: PostgreSQL 16+ with pgvector 0.8.0+
* **Containerization**: Docker & Docker Compose
  * PostgreSQL with pgvector in container
  * Application container with hot-reload
  * Volume mounts for development
  * Centralized logging
* **Runtime**: Python 3.11+ in Docker container

---

### 6. **Non-Functional Requirements**

* Local-only execution for privacy and speed
* Containerized deployment for consistency and isolation
* Requires internet connection for both OCR and LLM APIs
* Consider caching extracted questions to minimize API costs
* OCR processing speed: up to 2,000 pages/minute
* UI must handle 1000+ questions efficiently with pagination
* Response time <200ms for UI interactions
* Auto-save within 1 second of edit
* Container resource limits: 2GB RAM, 2 CPU cores
* Persistent volumes for uploads and database
* Health check endpoints for container orchestration
* Structured JSON logging for debugging

---

### 7. **Stretch Features** (Post-MVP)

* Tagging assistance using LLM
* Export to CSV or Google Sheets
* Duplicate detection
* Admin login for multiple reviewers

---

### 8. **Assumptions**

* PDFs may contain a mix of question types and inconsistent formatting
* Mistral OCR will accurately extract complex elements (tables, equations, diagrams)
* OCR outputs structured Markdown preserving document hierarchy
* Gemini 2.5 Flash Lite's 1M token window can handle entire exam papers
* Structured output guarantees consistent JSON format
* PDF files must be under 50MB and 1,000 pages per document
* Users will review and approve questions before final storage

---

### 9. **Milestones**

| Milestone                              | Date   |
| -------------------------------------- | ------ |
| Docker setup & API configuration       | Week 1 |
| Mistral OCR API integration            | Week 1 |
| Gemini 2.5 Flash Lite integration      | Week 2 |
| FastAPI endpoints & database setup     | Week 3 |
| Tabulator UI implementation            | Week 3 |
| End-to-end testing                     | Week 4 |
| Performance optimization & deployment  | Week 4 |

---

### 10. **Success Criteria**

* ≥ 90% of valid questions are extracted from supported PDFs
  * Leveraging Mistral OCR's ~94.9% accuracy rate
  * Enhanced by Gemini's structured output capabilities
* < 5% false positives in extraction
* 100% accurate mapping of questions to source PDF
* Approved questions reliably stored in DB
* UI response time <200ms for all interactions
* Zero data loss during review process

---

### 11. **Mistral OCR API Implementation Details**

#### API Integration
```python
from mistralai import Mistral
import os

# Initialize client
api_key = os.environ["MISTRAL_API_KEY"]
client = Mistral(api_key=api_key)

# Process PDF
ocr_response = client.ocr.process(
    model="mistral-ocr-latest",
    document={
        "type": "document_url",  # or "document_base64"
        "document_url": "path/to/exam_paper.pdf"
    },
    include_image_base64=True  # Include extracted images
)
```

#### Key Capabilities Utilized
* **Document Understanding**: Extract questions with preserved formatting
* **Table Recognition**: Capture mark schemes and scoring tables
* **Equation Handling**: Maintain mathematical expressions in LaTeX format
* **Image Extraction**: Preserve diagrams and figures with bounding boxes
* **Multilingual Support**: Handle papers in various languages

#### Error Handling
* Implement retry logic for API failures
* Fallback to batch processing for large files
* Cache processed PDFs to avoid re-processing

---

### 12. **Gemini 2.5 Flash Implementation Details**

#### Structured Output Schema
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

class Question(BaseModel):
    question_number: str = Field(..., description="Question number")
    marks: int = Field(..., description="Marks allocated")
    question_text: str = Field(..., description="Complete question text")
    topics: List[str] = Field(..., description="Topics covered")
    question_type: QuestionType = Field(..., description="Type of question")
    
class ExamPaper(BaseModel):
    year: str = Field(..., description="Year of exam")
    level: str = Field(..., description="Education level")
    source_pdf: str = Field(..., description="Source PDF filename")
    questions: List[Question] = Field(..., description="Extracted questions")
```

#### Smart Processing Pipeline
```python
async def extract_questions(markdown_text: str, pdf_filename: str):
    client = genai.Client(api_key=os.getenv('GOOGLE_API_KEY'))
    model = "gemini-2.5-flash"  # Stable version for structured extraction
    
    response = await client.models.generate_content_async(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=ExamPaper.model_json_schema(),
            temperature=0.1,
            max_output_tokens=8192
        )
    )
    
    return ExamPaper.model_validate_json(response.text)
```

#### Cost Optimization
* Process multiple questions in single API calls
* Cache extracted questions to avoid reprocessing
* Use thinking budgets efficiently
* Batch process similar documents

#### Total Cost Estimate (per 1000 exam papers)
* OCR: $50-100 (50-100 pages per paper average)
* LLM: $5-10 (assuming 10k tokens per paper)
* Total: ~$55-110 per 1000 papers

---

### 13. **Gemini Embedding Implementation**

```python
# services/embedding_service.py
from google import genai
from google.genai import types
import numpy as np
from typing import List, Dict, Optional

class GeminiEmbeddingService:
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)
        self.model = "models/embedding-001"  # Note: models/ prefix required
        self.dimension = 3072  # Full dimensions (can reduce with output_dimensionality)
        
    async def generate_embedding(self, text: str, task_type: str = "RETRIEVAL_DOCUMENT") -> List[float]:
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
        return response.embeddings[0].values
    
    async def generate_batch_embeddings(self, questions: List[Dict]) -> Dict[int, List[float]]:
        """Generate embeddings for multiple questions"""
        embeddings = {}
        
        for question in questions:
            # Combine question text with metadata for richer embeddings
            text_to_embed = f"""
            Question: {question['question_text']}
            Type: {question['question_type']}
            Topics: {', '.join(question['topics'])}
            Level: {question['level']}
            Marks: {question['marks']}
            """
            
            embedding = await self.generate_embedding(text_to_embed)
            embeddings[question['id']] = embedding
            
        return embeddings
```

```python
# database/vector_operations.py
from sqlalchemy import text
import json

class VectorOperations:
    def __init__(self, db_session):
        self.db = db_session
    
    def store_embedding(self, question_id: int, embedding: List[float], model_version: str = "1.0"):
        """Store embedding in PostgreSQL with pgvector"""
        query = text("""
            INSERT INTO question_embeddings (question_id, embedding, model_version)
            VALUES (:question_id, :embedding, :model_version)
            ON CONFLICT (question_id, model_name, model_version) 
            DO UPDATE SET 
                embedding = EXCLUDED.embedding,
                embedding_generated_at = CURRENT_TIMESTAMP
        """)
        
        self.db.execute(query, {
            'question_id': question_id,
            'embedding': json.dumps(embedding),
            'model_version': model_version
        })
        self.db.commit()
    
    def find_similar_questions(self, embedding: List[float], limit: int = 10, threshold: float = 0.8):
        """Find similar questions using cosine similarity"""
        query = text("""
            SELECT 
                q.id,
                q.question_text,
                q.topics,
                q.question_type,
                1 - (qe.embedding <=> :embedding::vector) AS similarity
            FROM questions q
            JOIN question_embeddings qe ON q.id = qe.question_id
            WHERE 1 - (qe.embedding <=> :embedding::vector) > :threshold
            ORDER BY qe.embedding <=> :embedding::vector
            LIMIT :limit
        """)
        
        return self.db.execute(query, {
            'embedding': json.dumps(embedding),
            'threshold': threshold,
            'limit': limit
        }).fetchall()
```

---

### 14. **PostgreSQL Configuration**

```python
# config.py
import os

class Config:
    # PostgreSQL connection
    POSTGRES_USER = os.getenv('POSTGRES_USER', 'postgres')
    POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD', 'password')
    POSTGRES_HOST = os.getenv('POSTGRES_HOST', 'localhost')
    POSTGRES_PORT = os.getenv('POSTGRES_PORT', '5432')
    POSTGRES_DB = os.getenv('POSTGRES_DB', 'question_bank')
    
    SQLALCHEMY_DATABASE_URI = (
        f'postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@'
        f'{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}'
    )
    
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Pool configuration for better performance
    SQLALCHEMY_ENGINE_OPTIONS = {
        'pool_size': 10,
        'pool_recycle': 3600,
        'pool_pre_ping': True,
        'max_overflow': 20
    }
    
    # API Keys
    MISTRAL_API_KEY = os.getenv('MISTRAL_API_KEY')
    GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
```

```python
# database/init_db.py
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

def init_postgresql():
    """Initialize PostgreSQL with pgvector extension"""
    # Connect to PostgreSQL server
    conn = psycopg2.connect(
        host=Config.POSTGRES_HOST,
        port=Config.POSTGRES_PORT,
        user=Config.POSTGRES_USER,
        password=Config.POSTGRES_PASSWORD
    )
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cur = conn.cursor()
    
    # Create database if not exists
    cur.execute(f"SELECT 1 FROM pg_database WHERE datname = '{Config.POSTGRES_DB}'")
    if not cur.fetchone():
        cur.execute(f"CREATE DATABASE {Config.POSTGRES_DB}")
    
    cur.close()
    conn.close()
    
    # Connect to the new database and create extensions
    conn = psycopg2.connect(
        host=Config.POSTGRES_HOST,
        port=Config.POSTGRES_PORT,
        database=Config.POSTGRES_DB,
        user=Config.POSTGRES_USER,
        password=Config.POSTGRES_PASSWORD
    )
    cur = conn.cursor()
    
    # Create extensions
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
    cur.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
    
    conn.commit()
    cur.close()
    conn.close()
```

---

### 15. **MCP (Model Context Protocol) Integration**

```json
// claude_desktop_config.json configuration
{
  "mcpServers": {
    "postgres-questions": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-postgres",
        "postgresql://readonly_user:password@localhost:5432/question_bank"
      ]
    }
  }
}
```

```sql
-- Create read-only user for MCP access
CREATE ROLE readonly_user WITH LOGIN PASSWORD 'secure_password';
GRANT CONNECT ON DATABASE question_bank TO readonly_user;
GRANT USAGE ON SCHEMA public TO readonly_user;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO readonly_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO readonly_user;
```

```python
# mcp_queries.py - Example queries for LLM usage
EXAMPLE_MCP_QUERIES = {
    "find_similar": """
        -- Find questions similar to a given topic
        SELECT q.*, 
               1 - (qe.embedding <=> (
                   SELECT embedding FROM question_embeddings 
                   WHERE question_id = :target_id
               )) as similarity
        FROM questions q
        JOIN question_embeddings qe ON q.id = qe.question_id
        WHERE q.id != :target_id
        ORDER BY similarity DESC
        LIMIT 10;
    """,
    
    "search_by_topic": """
        -- Search questions by topic with vector similarity
        SELECT DISTINCT q.*, 
               ts_rank(to_tsvector('english', q.question_text), 
                      plainto_tsquery('english', :search_term)) as text_rank
        FROM questions q
        WHERE :topic = ANY(q.topics)
        OR to_tsvector('english', q.question_text) @@ plainto_tsquery('english', :search_term)
        ORDER BY text_rank DESC
        LIMIT 20;
    """,
    
    "analytics": """
        -- Question distribution analytics
        SELECT 
            question_type,
            year,
            COUNT(*) as count,
            AVG(marks) as avg_marks,
            array_agg(DISTINCT unnest(topics)) as all_topics
        FROM questions
        GROUP BY question_type, year
        ORDER BY year DESC, count DESC;
    """
}
```

#### MCP Benefits for Question Bank:
1. **Natural Language Queries**: LLMs can query the database using natural language
2. **Semantic Search**: Find similar questions using vector embeddings
3. **Complex Analytics**: Generate insights about question patterns and distributions
4. **Read-Only Safety**: MCP enforces read-only access for data protection

---

### 16. **FastAPI Application Structure**

```
pdf_question_extractor/
├── app.py                 # Main FastAPI application
├── config.py              # Configuration settings
├── pyproject.toml         # Poetry dependencies
├── poetry.lock            # Lock file
├── .env                   # API keys (not in git)
├── .env.example           # Example environment vars
├── .gitignore
├── README.md
├── Dockerfile             # Container definition
├── docker-compose.yml     # Multi-container setup
├── database/
│   ├── __init__.py
│   ├── models.py         # SQLAlchemy async models
│   ├── session.py        # Async database sessions
│   ├── schema.sql        # Database schema
│   └── migrations/       # Alembic migrations
├── services/
│   ├── __init__.py
│   ├── ocr_service.py    # Mistral OCR integration
│   ├── llm_service.py    # Gemini integration
│   ├── embedding_service.py # Vector embeddings
│   └── pdf_processor.py  # PDF handling logic
├── api/
│   ├── __init__.py
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── questions.py  # Question endpoints
│   │   ├── upload.py     # Upload endpoints
│   │   └── health.py     # Health checks
│   ├── schemas/          # Pydantic models
│   │   ├── __init__.py
│   │   ├── question.py
│   │   └── response.py
│   └── dependencies.py   # Dependency injection
├── static/
│   ├── css/
│   │   └── style.css     # Custom styles
│   ├── js/
│   │   └── app.js        # Frontend logic
│   └── index.html        # Main UI
├── uploads/              # Temporary PDF storage
├── logs/                 # Application logs
└── tests/
    ├── __init__.py
    ├── conftest.py       # Pytest fixtures
    └── test_*.py         # Test modules
```

---

### 17. **API Endpoints**

```python
# FastAPI Routes with Async Support
POST   /api/upload          # Upload PDF folder (async)
GET    /api/process/{id}    # Process status (WebSocket available)
GET    /api/questions       # Get paginated questions
PUT    /api/questions/{id}  # Update single question
POST   /api/questions/bulk  # Bulk operations
POST   /api/questions/save  # Save to permanent DB (async)
GET    /api/export          # Export to CSV/JSON
GET    /api/health          # Health check for Docker
GET    /docs                # Auto-generated OpenAPI docs
GET    /redoc               # Alternative API documentation
WS     /ws/processing       # WebSocket for real-time updates
```

---

### 18. **Complete UI Implementation**

#### HTML Structure (static/index.html)
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>PDF Question Extractor</title>
    <link href="https://unpkg.com/tabulator-tables@5.5/dist/css/tabulator.min.css" rel="stylesheet">
    <link rel="stylesheet" href="/static/css/style.css">
</head>
<body>
    <div class="container">
        <h1>PDF Question Extractor</h1>
        
        <!-- Upload Section -->
        <div class="upload-section">
            <input type="file" id="pdfFiles" multiple accept=".pdf" webkitdirectory>
            <button onclick="uploadPDFs()">Process PDFs</button>
            <div id="uploadProgress"></div>
        </div>
        
        <!-- Review Section -->
        <div class="review-section">
            <div class="controls">
                <button onclick="approveAll()">Approve All</button>
                <button onclick="rejectAll()">Reject All</button>
                <button onclick="saveApproved()" class="save-btn">Save Approved</button>
                <button onclick="exportData()">Export CSV</button>
            </div>
            
            <div id="questionTable"></div>
            
            <div class="status-bar">
                <span id="statusText">0 questions loaded</span>
                <span id="saveStatus"></span>
            </div>
        </div>
    </div>
    
    <script src="https://unpkg.com/tabulator-tables@5.5/dist/js/tabulator.min.js"></script>
    <script src="/static/js/app.js"></script>
</body>
</html>
```

#### JavaScript Implementation (static/js/app.js)
```javascript
let table;
let autoSaveTimer;

// Initialize Tabulator
function initTable() {
    table = new Tabulator("#questionTable", {
        height: "600px",
        layout: "fitColumns",
        pagination: "local",
        paginationSize: 20,
        movableColumns: true,
        columns: [
            {title: "✓", field: "approved", formatter: "tickCross", 
             editor: true, width: 50, hozAlign: "center"},
            {title: "Q#", field: "question_number", editor: "input", width: 80},
            {title: "Question", field: "question_text", editor: "textarea", 
             formatter: "textarea", minWidth: 300},
            {title: "Marks", field: "marks", editor: "number", width: 80},
            {title: "Topics", field: "topics", editor: "input", 
             formatter: (cell) => cell.getValue().join(", ")},
            {title: "Type", field: "question_type", editor: "select",
             editorParams: {values: ["MCQ", "Essay", "Short Answer"]}, width: 120},
            {title: "Year", field: "year", editor: "input", width: 80},
            {title: "Level", field: "level", editor: "input", width: 100},
            {title: "Source", field: "source_pdf", width: 150}
        ],
        cellEdited: function(cell) {
            markModified(cell.getRow());
            scheduleAutoSave();
        }
    });
}

// API Functions
async function uploadPDFs() {
    const files = document.getElementById('pdfFiles').files;
    const formData = new FormData();
    
    for (let file of files) {
        formData.append('pdfs', file);
    }
    
    const response = await fetch('/api/upload', {
        method: 'POST',
        body: formData
    });
    
    const result = await response.json();
    loadQuestions();
}

async function loadQuestions() {
    const response = await fetch('/api/questions');
    const data = await response.json();
    table.setData(data.questions);
    updateStatus(`${data.total} questions loaded`);
}

async function saveQuestion(questionData) {
    try {
        await fetch(`/api/questions/${questionData.id}`, {
            method: 'PUT',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(questionData)
        });
    } catch (error) {
        console.error('Save failed:', error);
    }
}

async function saveApproved() {
    const approved = table.getData().filter(q => q.approved);
    
    const response = await fetch('/api/questions/save', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({questions: approved})
    });
    
    if (response.ok) {
        alert(`${approved.length} questions saved to database!`);
        loadQuestions(); // Reload to show updated status
    }
}

// UI Functions
function markModified(row) {
    row.getElement().classList.add("modified");
}

function scheduleAutoSave() {
    clearTimeout(autoSaveTimer);
    autoSaveTimer = setTimeout(() => {
        const modified = table.getData().filter(q => q.modified);
        modified.forEach(saveQuestion);
        updateStatus("", "Auto-saved");
    }, 1000);
}

function updateStatus(main, save = "") {
    document.getElementById('statusText').textContent = main;
    document.getElementById('saveStatus').textContent = save;
}

// Initialize on load
document.addEventListener('DOMContentLoaded', initTable);
```

---

### 19. **Processing Pipeline**

```python
# services/pdf_processor.py
import asyncio
from pathlib import Path

class PDFProcessor:
    def __init__(self, ocr_service, llm_service, embedding_service):
        self.ocr = ocr_service
        self.llm = llm_service
        self.embedding = embedding_service
    
    async def process_folder(self, folder_path):
        """Process all PDFs in folder"""
        pdf_files = list(Path(folder_path).glob("*.pdf"))
        results = []
        
        for pdf in pdf_files:
            try:
                # 1. OCR Processing
                markdown = await self.ocr.process_pdf(pdf)
                
                # 2. LLM Extraction
                questions = await self.llm.extract_questions(
                    markdown, 
                    pdf.name
                )
                
                # 3. Store in temporary DB
                stored_questions = []
                for q in questions:
                    stored_q = self.store_extracted(q)
                    stored_questions.append(stored_q)
                    results.append(stored_q)
                
                # 4. Generate embeddings in batch
                embeddings = await self.embedding.generate_batch_embeddings(
                    stored_questions
                )
                
                # 5. Store embeddings
                for q_id, embedding in embeddings.items():
                    self.store_embedding(q_id, embedding)
                    
            except Exception as e:
                logging.error(f"Failed to process {pdf}: {e}")
                continue
                
        return results
```

---

### 20. **Requirements.txt**

```
# Core dependencies
flask==3.0.0
flask-cors==4.0.0
sqlalchemy==2.0.23
python-dotenv==1.0.0

# API clients
mistralai==0.0.8
google-generativeai==0.3.2

# Processing
langchain==0.1.0
pydantic==2.5.3
tenacity==8.2.3

# Database
psycopg2-binary==2.9.9  # PostgreSQL adapter
pgvector==0.2.4  # pgvector Python client

# Utilities
pathlib==1.0.1
aiofiles==23.2.1
```

---

### 21. **Running Instructions**

```bash
# 1. Install PostgreSQL and pgvector
# Ubuntu/Debian:
sudo apt update
sudo apt install postgresql postgresql-contrib
sudo apt install postgresql-16-pgvector

# macOS:
brew install postgresql@16
brew install pgvector

# 2. Setup PostgreSQL Database
sudo -u postgres psql
CREATE DATABASE question_bank;
CREATE USER questionuser WITH PASSWORD 'your_secure_password';
GRANT ALL PRIVILEGES ON DATABASE question_bank TO questionuser;
\q

# 3. Setup Python Environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 4. Configure Environment
cp .env.example .env
# Edit .env with your settings:
# MISTRAL_API_KEY=your_key
# GOOGLE_API_KEY=your_key
# POSTGRES_USER=questionuser
# POSTGRES_PASSWORD=your_secure_password
# POSTGRES_HOST=localhost
# POSTGRES_PORT=5432
# POSTGRES_DB=question_bank

# 5. Initialize Database
python -c "from database.init_db import init_postgresql; init_postgresql()"
python manage.py db upgrade  # Run migrations

# 6. Create Read-Only User for MCP
psql -U questionuser -d question_bank
CREATE ROLE readonly_user WITH LOGIN PASSWORD 'readonly_password';
GRANT CONNECT ON DATABASE question_bank TO readonly_user;
GRANT USAGE ON SCHEMA public TO readonly_user;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO readonly_user;
\q

# 7. Run Application
python app.py

# 8. Open Browser
# Navigate to http://localhost:5000
```

### 22. **PostgreSQL Performance Tuning**

```sql
-- Optimize PostgreSQL for vector operations
ALTER SYSTEM SET shared_buffers = '256MB';
ALTER SYSTEM SET effective_cache_size = '1GB';
ALTER SYSTEM SET maintenance_work_mem = '64MB';
ALTER SYSTEM SET random_page_cost = 1.1;  -- For SSD storage

-- Reload configuration
SELECT pg_reload_conf();

-- Vacuum and analyze after bulk inserts
VACUUM ANALYZE questions;
VACUUM ANALYZE question_embeddings;
```

---

### 23. **Docker Configuration**

#### Docker Architecture
```yaml
# docker-compose.yml structure
services:
  postgres:
    image: pgvector/pgvector:pg16
    environment:
      - POSTGRES_USER=${POSTGRES_USER}
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
      - POSTGRES_DB=${POSTGRES_DB}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./database/schema.sql:/docker-entrypoint-initdb.d/01-schema.sql
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER}"]
      interval: 10s
      timeout: 5s
      retries: 5

  app:
    build: 
      context: ./pdf_question_extractor
      dockerfile: Dockerfile
    environment:
      - DATABASE_URL=postgresql+asyncpg://${POSTGRES_USER}:${POSTGRES_PASSWORD}@postgres:5432/${POSTGRES_DB}
      - MISTRAL_API_KEY=${MISTRAL_API_KEY}
      - GOOGLE_API_KEY=${GOOGLE_API_KEY}
    volumes:
      - ./pdf_question_extractor:/app
      - ./uploads:/app/uploads
    ports:
      - "8000:8000"
    depends_on:
      postgres:
        condition: service_healthy
    command: uvicorn app:app --host 0.0.0.0 --port 8000 --reload

volumes:
  postgres_data:
```

#### Dockerfile Configuration
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY pyproject.toml poetry.lock ./
RUN pip install poetry && \
    poetry config virtualenvs.create false && \
    poetry install --no-interaction --no-ansi

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p uploads logs

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=40s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health')"

EXPOSE 8000
```

#### Development Benefits
1. **Consistent Environment**: Same setup across all developers
2. **Easy Onboarding**: Single `docker-compose up` command
3. **Database Management**: pgvector pre-installed and configured
4. **Hot Reload**: Code changes reflected immediately
5. **Isolated Dependencies**: No system pollution
6. **Better Logging**: Centralized log collection
7. **Easy Debugging**: Container logs and exec access

#### Container Security
- Non-root user execution
- Read-only root filesystem (production)
- Limited resource allocation
- No privileged mode
- Secrets via environment variables
- Network isolation between containers

---
