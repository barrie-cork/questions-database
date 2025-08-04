# Docker Quick Start Guide

Get the PDF Question Extractor running with Docker in minutes.

## Prerequisites
- Docker and Docker Compose installed ([Get Docker](https://docs.docker.com/get-docker/))
- API keys for Mistral and Google Gemini
- 4GB free RAM
- 2GB free disk space

## Quick Start

### 1. Clone and Setup

```bash
# Clone the repository
git clone [repository-url]
cd questions_pdf_to_sheet

# Setup environment
cd pdf_question_extractor
cp .env.example .env
```

### 2. Configure API Keys

Edit `.env` with your API keys:
```env
MISTRAL_API_KEY=your_mistral_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
```

### 3. Start Services

```bash
# Go back to root directory
cd ..

# Start everything
docker-compose up -d

# Or use the Makefile
make up
```

### 4. Verify Installation

```bash
# Check containers are running
docker-compose ps
# Should show both 'postgres' and 'app' containers as "Up"

# Check application health
curl http://localhost:8000/health
```

### 5. Access the Application

- **Web UI**: http://localhost:8000
- **API Documentation**: http://localhost:8000/api/docs
- **API Health Check**: http://localhost:8000/health

## Development Workflow

### Using the Makefile (Recommended)

```bash
make up          # Start services
make logs        # View logs
make shell       # Open shell in app container
make db-shell    # Open PostgreSQL shell
make test        # Run tests
make down        # Stop services
make clean       # Remove containers and volumes
make fresh       # Clean rebuild and start
```

### Manual Docker Commands

```bash
# View logs
docker-compose logs -f app
docker-compose logs -f postgres

# Run tests inside container
docker-compose exec app pytest

# Access app shell
docker-compose exec app bash

# Access database
docker-compose exec postgres psql -U questionuser -d question_bank

# Stop everything
docker-compose down

# Remove everything (including data)
docker-compose down -v
```

### Development Mode

For development with hot-reload and pgAdmin:

```bash
# Using docker-compose
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up

# Or using Makefile
make up-dev
```

Then access:
- pgAdmin: http://localhost:5050
  - Email: admin@admin.com
  - Password: admin

## Container Architecture

```
┌─────────────────────┐     ┌─────────────────────┐
│                     │     │                     │
│    App Container    │────▶│ PostgreSQL Container│
│   (FastAPI App)     │     │  (Database + pgvector)│
│                     │     │                     │
└─────────────────────┘     └─────────────────────┘
         │                            │
         │                            │
         ▼                            ▼
    Port 8000                    Port 5432
   (Web Interface)              (Database)
```

## Important Notes

### Volumes and Persistence
- **Database data**: Persists in Docker volumes
- **Uploaded files**: Stored in `./uploads` directory
- **Logs**: Available in `./logs` directory
- **Code changes**: Automatically reflected (volume mounted)

### Resource Usage
- **Memory**: ~2GB RAM recommended
- **CPU**: 2+ cores recommended
- **Disk**: ~1GB for Docker images + data

### Network Configuration
- App runs on port 8000
- PostgreSQL on port 5432 (exposed for development)
- All services on the same Docker network

## Troubleshooting

### Container Won't Start

```bash
# Check logs for errors
docker-compose logs app

# Common fixes:
# 1. Check .env file exists and has valid keys
# 2. Ensure ports 8000 and 5432 are free
# 3. Rebuild if needed
docker-compose build --no-cache app
docker-compose up -d
```

### Database Connection Issues

```bash
# Verify postgres is healthy
docker-compose ps
# Look for "postgres ... Up (healthy)"

# Check database logs
docker-compose logs postgres

# Test connection
docker-compose exec postgres pg_isready
```

### Permission Issues

```bash
# Fix file permissions if needed
chmod -R 755 ./uploads ./logs

# On Linux, you might need to set ownership
sudo chown -R $USER:$USER ./uploads ./logs
```

### Port Conflicts

If ports are already in use:

1. Stop conflicting services, or
2. Change ports in `docker-compose.yml`:
   ```yaml
   ports:
     - "8001:8000"  # Change 8001 to any free port
   ```

### Reset Everything

```bash
# Stop and remove all containers and volumes
make clean

# Fresh start with clean build
make fresh
```

## Docker Commands Reference

### Container Management
```bash
docker-compose up -d        # Start in background
docker-compose down         # Stop containers
docker-compose restart app  # Restart app container
docker-compose ps          # List containers
```

### Logs and Debugging
```bash
docker-compose logs -f app      # Follow app logs
docker-compose logs --tail=50   # Last 50 lines
docker-compose exec app bash    # Shell access
```

### Database Operations
```bash
# Backup database
docker-compose exec postgres pg_dump -U questionuser question_bank > backup.sql

# Restore database
docker-compose exec -T postgres psql -U questionuser question_bank < backup.sql
```

### Cleanup
```bash
docker system prune -a      # Remove unused images
docker volume prune         # Remove unused volumes
```

## Tips for Production

1. **Environment Variables**: Use `.env.production` with strong passwords
2. **Volumes**: Use named volumes for better data management
3. **Networking**: Put behind a reverse proxy (nginx/traefik)
4. **Monitoring**: Add health checks and logging aggregation
5. **Backups**: Implement automated database backups

## Next Steps

1. Upload your first PDF through the web UI
2. Check the API documentation at `/api/docs`
3. Monitor processing through the interface
4. Export your extracted questions

For more detailed documentation, see:
- [Main README](README.md)
- [API Reference](docs/API_REFERENCE.md)
- [Developer Guide](docs/DEVELOPER_QUICKSTART.md)