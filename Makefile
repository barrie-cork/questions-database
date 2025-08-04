.PHONY: help build up down logs shell db-shell test lint format clean

# Default target
help:
	@echo "PDF Question Extractor - Docker Commands"
	@echo "======================================="
	@echo "make build          - Build Docker images"
	@echo "make up             - Start all services"
	@echo "make up-dev         - Start with development config"
	@echo "make down           - Stop all services"
	@echo "make logs           - View container logs"
	@echo "make shell          - Open shell in app container"
	@echo "make db-shell       - Open PostgreSQL shell"
	@echo "make test           - Run tests"
	@echo "make lint           - Run code linting"
	@echo "make format         - Format code"
	@echo "make clean          - Remove containers and volumes"
	@echo "make fresh          - Clean and rebuild everything"

# Build Docker images
build:
	docker-compose build

# Start services
up:
	docker-compose up -d

# Start with development configuration
up-dev:
	docker-compose -f docker-compose.yml -f docker-compose.dev.yml up

# Stop services
down:
	docker-compose down

# View logs
logs:
	docker-compose logs -f

# Open shell in app container
shell:
	docker-compose exec app bash

# Open PostgreSQL shell
db-shell:
	docker-compose exec postgres psql -U ${POSTGRES_USER:-postgres} -d ${POSTGRES_DB:-question_bank}

# Run tests
test:
	docker-compose exec app pytest

# Run linting
lint:
	docker-compose exec app ruff check .
	docker-compose exec app mypy .

# Format code
format:
	docker-compose exec app black .
	docker-compose exec app ruff check --fix .

# Clean up
clean:
	docker-compose down -v
	docker system prune -f

# Fresh start
fresh: clean build up

# Database operations
db-migrate:
	docker-compose exec app alembic upgrade head

db-reset:
	docker-compose exec app alembic downgrade base
	docker-compose exec app alembic upgrade head

# Development shortcuts
dev: up-dev

stop: down

restart: down up