# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## PDF Question Extractor Project

FastAPI application extracting questions from exam PDFs using Mistral OCR and Google Gemini, with PostgreSQL + pgvector for semantic search.

### Quick Commands

```bash
# Development
cd pdf_question_extractor
python app.py          # Run locally (http://localhost:8000)
make up-dev           # Docker development mode
make test             # Run tests
make lint             # Check code quality

# Database
make db-migrate       # Apply migrations
make db-shell         # PostgreSQL shell
python database/init_db.py  # Initialize database
```

### Key Implementation Details

**Architecture**: PDFProcessor → MistralOCR → GeminiLLM → Embeddings → PostgreSQL/pgvector

**Service Structure**:
- `services/ocr_service.py` - Mistral OCR (max 1000 pages, 50MB)
- `services/llm_service.py` - Gemini extraction (gemini-2.0-flash-exp)
- `services/embedding_service.py` - Vector embeddings (768-dim)
- `services/pdf_processor.py` - Orchestrates pipeline

**Database Tables**:
- `extracted_questions` - Temporary review storage
- `questions` - Approved questions
- `question_embeddings` - Semantic search vectors

**API Patterns**:
- Async/await throughout
- WebSocket for real-time updates
- Rate limiting with exponential backoff
- Tenacity for retry logic

### Project Commands (SuperClaude)

**`/analyze-pdf`** - Analyze PDF structure and extract questions
**`/improve-extraction`** - Enhance OCR and question detection algorithms
**`/test-pipeline`** - Run end-to-end pipeline tests

### Active Personas

- **analyzer**: PDF structure analysis and question pattern detection
- **backend**: API development and database operations
- **frontend**: Web interface improvements
- **qa**: Testing OCR accuracy and extraction quality

### Project-Specific Rules

1. Always validate PDF format before processing (max 50MB, 1000 pages)
2. Use embedding service for semantic question matching
3. Maintain vector database consistency
4. Test OCR accuracy on sample PDFs before deployment
5. Handle rate limits gracefully (60 rpm Mistral, 15 rpm Gemini)
6. Process PDFs atomically - store questions and embeddings together

### MCP Server Usage

- **Sequential**: Complex PDF analysis workflows
- **Context7**: Documentation for PDF libraries and OCR tools
- **Playwright**: E2E testing of web upload interface