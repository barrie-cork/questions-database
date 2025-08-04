# PDF Question Extractor API Setup Guide

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Environment Setup
Create a `.env` file in the project root:
```env
# Database Configuration
POSTGRES_USER=questionuser
POSTGRES_PASSWORD=securepassword123
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=question_bank

# API Keys
MISTRAL_API_KEY=your_mistral_api_key_here
GOOGLE_API_KEY=your_google_api_key_here

# Application Settings
APP_ENV=development
LOG_LEVEL=INFO
MAX_UPLOAD_SIZE=52428800
```

### 3. Database Setup
```bash
# Start PostgreSQL with pgvector extension
# Install pgvector extension in your database
CREATE EXTENSION IF NOT EXISTS vector;

# Run database initialization
python -c "from database.init_db import main; main()"
```

### 4. Run the API
```bash
# Development mode with auto-reload
python app.py

# Or with uvicorn directly
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### 5. Test the API
```bash
# Run the test script
./test_api_endpoints.sh

# Or access the interactive docs
open http://localhost:8000/api/docs
```

## API Endpoints Overview

### Core Endpoints
- `GET /health` - Health check
- `POST /api/upload` - Upload PDF files
- `GET /api/process/{id}` - Processing status
- `GET /api/questions` - List questions (paginated)
- `PUT /api/questions/{id}` - Update question
- `POST /api/questions/bulk` - Bulk operations
- `POST /api/questions/save` - Save approved questions
- `GET /api/export` - Export questions
- `GET /api/stats` - System statistics
- `WebSocket /api/ws/processing` - Real-time updates

### File Operations
- `GET /api/download/{filename}` - Download exported files
- Upload directory: `/uploads/`
- Export directory: `/cache/exports/`

## Development Tips

### Testing Individual Endpoints
```bash
# Health check
curl http://localhost:8000/health

# Get questions
curl "http://localhost:8000/api/questions?page=1&per_page=5"

# Upload PDF
curl -X POST "http://localhost:8000/api/upload" \
  -F "files=@test.pdf" \
  -F "request_data={\"store_to_db\": true}"
```

### WebSocket Testing
```bash
# Install wscat
npm install -g wscat

# Connect to WebSocket
wscat -c ws://localhost:8000/api/ws/processing

# Send ping
{"type": "ping"}
```

### Database Queries
```sql
-- Check extracted questions
SELECT COUNT(*) FROM extracted_questions;

-- Check permanent questions  
SELECT COUNT(*) FROM questions;

-- Check question status distribution
SELECT status, COUNT(*) FROM extracted_questions GROUP BY status;
```

## Troubleshooting

### Common Issues

1. **Database Connection Error**
   - Ensure PostgreSQL is running
   - Check connection parameters in `.env`
   - Verify pgvector extension is installed

2. **Import Errors**
   - Install missing dependencies: `pip install -r requirements.txt`
   - Check Python version compatibility (3.8+)

3. **API Key Issues**
   - Verify MISTRAL_API_KEY and GOOGLE_API_KEY in `.env`
   - Check API key permissions and quotas

4. **File Upload Issues**
   - Ensure `/uploads/` directory exists and is writable
   - Check file size limits (default: 50MB)
   - Verify PDF file format

### Logs
Check application logs in `/logs/app.log` for detailed error information.

## Production Deployment

### Docker Setup
```bash
# Build image
docker build -t pdf-question-extractor .

# Run container
docker run -p 8000:8000 \
  -e POSTGRES_HOST=db_host \
  -e MISTRAL_API_KEY=your_key \
  -e GOOGLE_API_KEY=your_key \
  pdf-question-extractor
```

### Environment Variables
Set `APP_ENV=production` for production deployment to hide detailed error messages.

## API Documentation

The API includes interactive documentation:
- Swagger UI: http://localhost:8000/api/docs
- ReDoc: http://localhost:8000/api/redoc
- OpenAPI JSON: http://localhost:8000/api/openapi.json