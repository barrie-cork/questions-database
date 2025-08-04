# Developer Quick Start Guide

Get up and running with the PDF Question Extractor in under 10 minutes.

## 🚀 Quick Setup (5 minutes)

### Prerequisites
- Docker Desktop installed and running
- Git installed
- 4GB free RAM
- API keys ready (optional for initial exploration)

### 1. Clone and Navigate
```bash
git clone https://github.com/yourusername/pdf-question-extractor.git
cd pdf-question-extractor
```

### 2. Environment Setup
```bash
cd pdf_question_extractor
cp .env.example .env
# Edit .env to add your API keys (or skip for now)
```

### 3. Start Everything
```bash
cd ..  # Back to root directory
make up
```

### 4. Verify It's Working
- Web UI: http://localhost:8000
- API Docs: http://localhost:8000/api/docs
- Health Check: http://localhost:8000/health

## 🎯 First Tasks (15 minutes)

### Task 1: Upload and Process a PDF
1. Open http://localhost:8000
2. Click "Upload PDFs" or drag a PDF file
3. Watch real-time processing progress
4. Review extracted questions in the table

### Task 2: Explore the API
1. Visit http://localhost:8000/api/docs
2. Try the `/health` endpoint
3. Upload a PDF via the API:
```bash
curl -X POST "http://localhost:8000/api/upload" \
  -H "accept: application/json" \
  -F "pdfs=@sample.pdf"
```

### Task 3: Review Extracted Questions
1. Click on any question in the web UI
2. Edit the question text or metadata
3. Mark as "Approved" or "Rejected"
4. Save changes (auto-saves after 1 second)

## 🏗️ Understanding the Architecture

### Core Components
```
User → Web UI → FastAPI → Services → Database
                   ↓
              WebSocket (real-time updates)
```

### Service Layer
1. **OCR Service** - Extracts text from PDFs (Mistral)
2. **LLM Service** - Identifies questions (Gemini)
3. **Embedding Service** - Creates vectors (Gemini)
4. **PDF Processor** - Orchestrates everything

### Key Files to Review
```
pdf_question_extractor/
├── app.py              # Main FastAPI application
├── api/routes.py       # API endpoints
├── services/
│   ├── pdf_processor.py    # Main orchestrator
│   ├── ocr_service.py      # OCR integration
│   └── llm_service.py      # Question extraction
└── static/
    └── js/app.js       # Frontend logic
```

## 💻 Common Development Tasks

### View Logs
```bash
make logs        # All logs
make logs-app    # Just app logs
make logs-db     # Database logs
```

### Run Tests
```bash
make test        # Run all tests
make test-cov    # With coverage report
```

### Database Access
```bash
make db-shell    # PostgreSQL CLI
make db-backup   # Backup database
```

### Development Mode
```bash
make up-dev      # Hot reload enabled
```

## 🔧 Configuration

### Essential Environment Variables
```bash
# .env file
MISTRAL_API_KEY=your_key_here    # For OCR
GOOGLE_API_KEY=your_key_here     # For LLM & embeddings

# Optional
MAX_FILE_SIZE=52428800           # 50MB default
CONCURRENT_LIMIT=3               # Parallel PDFs
LOG_LEVEL=INFO                   # DEBUG for more detail
```

### Database Connection
Default connection string:
```
postgresql://questionuser:questionpass@postgres:5432/question_bank
```

## 🐛 Debugging Tips

### 1. Container Issues
```bash
make down        # Stop everything
make clean       # Remove volumes
make up          # Fresh start
```

### 2. API Errors
- Check logs: `make logs-app`
- Verify API keys in `.env`
- Test health endpoint: http://localhost:8000/health

### 3. Database Issues
```bash
make db-reset    # Reset database
make db-init     # Reinitialize schema
```

### 4. Processing Failures
- Check file size (< 50MB)
- Verify PDF is valid
- Monitor WebSocket messages in browser console

## 📚 Next Steps

### Learn the Codebase
1. Read [Services Documentation](SERVICES_DOCUMENTATION.md)
2. Review [API Design](API_DESIGN.md)
3. Understand [Database Design](DATABASE_DESIGN.md)

### Make Your First Contribution
1. Pick a task from [Junior Developer Task List](JUNIOR_DEVELOPER_TASK_LIST.md)
2. Create a feature branch
3. Make changes and test
4. Submit a pull request

### Explore Advanced Features
1. Try batch processing multiple PDFs
2. Test the search functionality
3. Export questions to CSV
4. Implement a new question type

## 🎓 Learning Resources

### Code Examples
- [Upload PDF](../pdf_question_extractor/tests/test_api_endpoints.py)
- [Process Questions](../pdf_question_extractor/services/pdf_processor.py)
- [WebSocket Client](../pdf_question_extractor/static/js/app.js)

### Key Concepts
- **Async/Await**: All I/O operations are async
- **Pydantic Models**: Type-safe request/response
- **SQLAlchemy**: Async ORM for database
- **WebSockets**: Real-time progress updates

## 🆘 Getting Help

### Documentation
- Can't find something? Check [PROJECT_INDEX.md](PROJECT_INDEX.md)
- API questions? See interactive docs at `/api/docs`
- Architecture questions? Read [System Architecture](SYSTEM_ARCHITECTURE.md)

### Debugging
1. Always check logs first: `make logs`
2. Verify your `.env` configuration
3. Test with the provided sample PDFs
4. Use the health check endpoint

### Common Issues

**"No API key" errors**
- Solution: Add keys to `.env` file

**"Connection refused" errors**
- Solution: Ensure Docker is running

**"Out of memory" errors**
- Solution: Increase Docker memory allocation

**Processing seems stuck**
- Solution: Check logs for rate limiting

## 🎉 You're Ready!

You now have:
- ✅ A running PDF Question Extractor
- ✅ Understanding of basic operations
- ✅ Knowledge of where to find help
- ✅ Tools to start developing

**Next**: Pick a task from the [Junior Developer Task List](JUNIOR_DEVELOPER_TASK_LIST.md) and start coding!

---

*Need more help? Check the [full documentation](README.md) or ask in the team chat.*