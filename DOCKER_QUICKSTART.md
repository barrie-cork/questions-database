# Docker Quick Start Guide

## Prerequisites
- Docker and Docker Compose installed
- super_c virtual environment activated (for running docker commands)
- API keys for Mistral and Google Gemini

## Quick Start

1. **Activate super_c environment**:
   ```bash
   cd /mnt/d/Python/Projects/Dave
   source super_c/bin/activate
   cd questions_pdf_to_sheet
   ```

2. **Setup environment**:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and database password
   ```

3. **Start services**:
   ```bash
   # Using docker-compose directly
   docker-compose up -d
   
   # Or using the Makefile
   make up
   ```

4. **Verify everything is running**:
   ```bash
   docker-compose ps
   # Should show both postgres and app containers as "Up"
   ```

5. **Access the application**:
   - API Documentation: http://localhost:8000/docs
   - Application: http://localhost:8000
   - pgAdmin (if using dev setup): http://localhost:5050

## Development Workflow

### Using the Makefile (Recommended)
```bash
make up          # Start services
make logs        # View logs
make shell       # Open shell in app container
make db-shell    # Open PostgreSQL shell
make test        # Run tests
make down        # Stop services
```

### Manual Docker Commands
```bash
# View logs
docker-compose logs -f app

# Run tests inside container
docker-compose exec app pytest

# Access app shell
docker-compose exec app bash

# Access database
docker-compose exec postgres psql -U postgres -d question_bank
```

### Development Mode
For development with hot-reload and pgAdmin:
```bash
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up
# Or
make up-dev
```

## Important Notes

1. **Virtual Environment**: The super_c virtual environment is used only for running Docker commands. The actual Python application runs inside the Docker container with its own isolated environment.

2. **File Changes**: Code changes are automatically reflected in the container due to volume mounting (hot-reload enabled).

3. **Database**: PostgreSQL with pgvector runs in a separate container. Data persists in Docker volumes even after stopping containers.

4. **Logs**: Application logs are available in the `./logs` directory and via `docker-compose logs`.

## Troubleshooting

### Container won't start
```bash
# Check logs
docker-compose logs app

# Rebuild if needed
docker-compose build --no-cache app
docker-compose up -d
```

### Database connection issues
```bash
# Verify postgres is healthy
docker-compose ps
# Should show postgres as "Up (healthy)"

# Check database logs
docker-compose logs postgres
```

### Permission issues
```bash
# Fix ownership if needed (from host)
sudo chown -R 1000:1000 ./uploads ./logs
```

### Reset everything
```bash
make clean  # Remove containers and volumes
make fresh  # Clean rebuild and start
```