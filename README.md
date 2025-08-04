# PDF Question Extractor

An intelligent system that extracts exam questions from PDF files using AI-powered OCR and natural language processing. The system provides a complete pipeline from PDF processing to searchable question database with vector embeddings for semantic search.

## Features

- **AI-Powered OCR**: Extracts text from PDF files using Mistral's Pixtral OCR model
- **Intelligent Question Extraction**: Uses Google Gemini to identify and structure exam questions
- **Semantic Search**: Vector embeddings enable finding similar questions
- **Real-time Progress Tracking**: WebSocket support for live processing updates
- **Web Interface**: Modern, responsive UI for reviewing and editing extracted questions
- **Batch Processing**: Process multiple PDFs concurrently with configurable limits
- **Export Functionality**: Export approved questions in CSV or JSON format

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Web UI (FastAPI + Vanilla JS)                │
│                         WebSocket Support                        │
├─────────────────────────────────────────────────────────────────┤
│                        PDF Processor                             │
│              (Orchestration & Progress Tracking)                 │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────┐   │
│  │ OCR Service  │  │ LLM Service  │  │ Embedding Service  │   │
│  │  (Mistral)   │  │   (Gemini)   │  │    (Gemini)        │   │
│  └──────────────┘  └──────────────┘  └────────────────────┘   │
├─────────────────────────────────────────────────────────────────┤
│                     PostgreSQL + pgvector                        │
│                  (Question Storage & Search)                     │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

### Using Docker (Recommended)

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/pdf-question-extractor.git
   cd pdf-question-extractor
   ```

2. **Set up environment variables**:
   ```bash
   cd pdf_question_extractor
   cp .env.example .env
   # Edit .env with your API keys
   ```

3. **Start the application**:
   ```bash
   make up
   ```

4. **Access the application**:
   - Web UI: http://localhost:8000
   - API Documentation: http://localhost:8000/api/docs

### Manual Installation

1. **Prerequisites**:
   - Python 3.11+
   - PostgreSQL 16+ with pgvector extension
   - API Keys for Mistral and Google Gemini

2. **Install dependencies**:
   ```bash
   cd pdf_question_extractor
   pip install -r requirements.txt
   ```

3. **Set up database**:
   ```bash
   # Create database and install extensions
   createdb question_bank
   psql question_bank -c "CREATE EXTENSION IF NOT EXISTS vector;"
   psql question_bank -c "CREATE EXTENSION IF NOT EXISTS pg_trgm;"
   
   # Run schema
   psql question_bank -f database/schema.sql
   ```

4. **Configure environment**:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

5. **Run the application**:
   ```bash
   python app.py
   ```

## Usage

### Web Interface

1. **Upload PDF**: Drag and drop or click to upload exam PDFs
2. **Monitor Progress**: Real-time updates show OCR, extraction, and storage progress
3. **Review Questions**: Edit extracted questions, marks, topics, and metadata
4. **Approve/Reject**: Mark questions for inclusion in final database
5. **Export**: Download approved questions as CSV or JSON

### API Endpoints

- `POST /api/upload` - Upload PDF for processing
- `GET /api/questions` - List extracted questions (paginated)
- `PUT /api/questions/{id}` - Update question details
- `POST /api/questions/bulk` - Bulk operations (approve/reject/delete)
- `GET /api/export` - Export questions in various formats
- `WS /ws/processing` - WebSocket for real-time progress

### Command Line

```python
from services.pdf_processor import process_pdf_file

# Process single PDF
result = await process_pdf_file("exam.pdf")

# Process directory
from services.pdf_processor import process_pdf_directory
results = await process_pdf_directory("exams/", recursive=True)
```

## Configuration

### Environment Variables

```bash
# API Keys (Required)
MISTRAL_API_KEY=your_mistral_api_key
GOOGLE_API_KEY=your_google_api_key

# Database Configuration
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=question_bank
POSTGRES_USER=questionuser
POSTGRES_PASSWORD=your_password

# Processing Configuration
MAX_FILE_SIZE=52428800  # 50MB
CONCURRENT_LIMIT=3      # Max parallel PDFs
RATE_LIMIT=60          # API calls per minute
```

### Service Configuration

- **OCR**: Mistral Pixtral model with 50MB file limit
- **LLM**: Gemini 2.5 Flash with structured output
- **Embeddings**: 768-dimensional vectors for semantic search
- **Rate Limiting**: Automatic API quota management

## Development

### Project Structure

```
pdf_question_extractor/
├── app.py                 # FastAPI application
├── api/                   # API routes and schemas
├── database/              # Database models and operations
├── services/              # Core business logic
│   ├── ocr_service.py    # Mistral OCR integration
│   ├── llm_service.py    # Gemini question extraction
│   ├── embedding_service.py # Vector generation
│   └── pdf_processor.py  # Pipeline orchestration
├── static/               # Web UI assets
└── tests/               # Test suite
```

### Running Tests

```bash
# Run all tests
make test

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test file
pytest tests/test_services.py -v
```

### Development Mode

```bash
# Start with hot reload
make up-dev

# View logs
make logs

# Access database
make db-shell
```

## Features in Detail

### OCR Processing
- Supports local files and URLs
- Automatic retry with exponential backoff
- Robust response parsing with fallbacks
- Progress tracking at each stage

### Question Extraction
- 8 question types supported (MCQ, Essay, Short Answer, etc.)
- Preserves question numbering and structure
- Handles multi-part questions
- Smart document chunking for large files

### Vector Search
- Semantic similarity search
- Duplicate detection
- Topic-based filtering
- Efficient batch processing

### Web Interface
- Drag-and-drop file upload
- Real-time progress via WebSocket
- Auto-save with debouncing
- Bulk operations support
- Responsive design

## API Documentation

Full API documentation is available at http://localhost:8000/api/docs when running the application.

### Example: Upload and Process PDF

```bash
curl -X POST "http://localhost:8000/api/upload" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@exam.pdf"
```

### Example: Search Questions

```bash
curl -X GET "http://localhost:8000/api/questions?search=calculus&page=1&per_page=20" \
  -H "accept: application/json"
```

## Monitoring

### Health Check

```bash
curl http://localhost:8000/api/health
```

### Metrics
- Processing success/failure rates
- Average questions per PDF
- API response times
- Resource utilization

## Security

- API keys stored in environment variables
- Input validation and sanitization
- Rate limiting to prevent abuse
- SQL injection prevention
- File size and type restrictions

## Troubleshooting

### Common Issues

1. **Database Connection Failed**
   - Ensure PostgreSQL is running
   - Check connection parameters in .env
   - Verify pgvector extension is installed

2. **API Rate Limits**
   - System automatically handles rate limiting
   - Adjust RATE_LIMIT in configuration
   - Check API quota status

3. **Memory Issues**
   - Reduce CONCURRENT_LIMIT for large PDFs
   - Increase Docker memory allocation
   - Process files in smaller batches

### Logs

```bash
# View application logs
docker-compose logs -f app

# View database logs
docker-compose logs -f postgres
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Mistral AI for OCR capabilities
- Google Gemini for question extraction
- pgvector for vector similarity search
- FastAPI for the web framework

## Documentation

- [Services Documentation](docs/SERVICES_DOCUMENTATION.md) - Detailed service layer documentation
- [API Reference](http://localhost:8000/api/docs) - Interactive API documentation
- [Development Guide](docs/DEVELOPMENT.md) - Development setup and guidelines
- [Docker Guide](DOCKER_QUICKSTART.md) - Docker setup and commands