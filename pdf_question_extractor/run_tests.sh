#!/bin/bash

# PDF Question Extractor - Test Runner Script
# This script runs the full test suite with various options

echo "========================================="
echo "PDF Question Extractor - Test Suite"
echo "========================================="

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running in Docker
if [ -f /.dockerenv ]; then
    echo -e "${GREEN}Running in Docker container${NC}"
    IN_DOCKER=true
else
    echo -e "${YELLOW}Running outside Docker - using docker-compose exec${NC}"
    IN_DOCKER=false
fi

# Function to run tests
run_tests() {
    local test_command="$1"
    local description="$2"
    
    echo -e "\n${YELLOW}Running: ${description}${NC}"
    echo "Command: $test_command"
    echo "-----------------------------------------"
    
    if [ "$IN_DOCKER" = true ]; then
        eval "$test_command"
    else
        docker-compose exec app $test_command
    fi
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ ${description} passed${NC}"
    else
        echo -e "${RED}✗ ${description} failed${NC}"
    fi
}

# Parse command line arguments
TEST_TYPE="${1:-all}"

case $TEST_TYPE in
    "unit")
        run_tests "pytest tests/test_ocr_service.py tests/test_llm_service.py tests/test_embedding_service.py -v" "Unit Tests"
        ;;
    "integration")
        run_tests "pytest tests/test_integration.py -v" "Integration Tests"
        ;;
    "api")
        run_tests "pytest tests/test_api.py -v" "API Tests"
        ;;
    "websocket")
        run_tests "pytest tests/test_websocket.py -v" "WebSocket Tests"
        ;;
    "coverage")
        run_tests "pytest --cov=. --cov-report=html --cov-report=term-missing" "Tests with Coverage"
        ;;
    "quick")
        run_tests "pytest -x --tb=short" "Quick Test Run (stop on first failure)"
        ;;
    "verbose")
        run_tests "pytest -vv --tb=long" "Verbose Test Run"
        ;;
    "all")
        echo -e "${GREEN}Running full test suite...${NC}\n"
        
        # Run each test category
        run_tests "pytest tests/test_ocr_service.py -v -m unit" "OCR Service Tests"
        run_tests "pytest tests/test_llm_service.py -v -m unit" "LLM Service Tests"
        run_tests "pytest tests/test_embedding_service.py -v -m unit" "Embedding Service Tests"
        run_tests "pytest tests/test_integration.py -v -m integration" "Integration Tests"
        run_tests "pytest tests/test_api.py -v -m unit" "API Tests"
        run_tests "pytest tests/test_websocket.py -v -m unit" "WebSocket Tests"
        
        # Final coverage report
        echo -e "\n${YELLOW}Generating coverage report...${NC}"
        run_tests "pytest --cov=. --cov-report=html --cov-report=term" "Coverage Report"
        ;;
    *)
        echo "Usage: $0 [unit|integration|api|websocket|coverage|quick|verbose|all]"
        echo ""
        echo "Options:"
        echo "  unit        - Run unit tests only"
        echo "  integration - Run integration tests only"
        echo "  api         - Run API tests only"
        echo "  websocket   - Run WebSocket tests only"
        echo "  coverage    - Run all tests with coverage report"
        echo "  quick       - Quick run, stop on first failure"
        echo "  verbose     - Verbose output with full tracebacks"
        echo "  all         - Run all test categories (default)"
        exit 1
        ;;
esac

echo -e "\n========================================="
echo -e "${GREEN}Test run completed!${NC}"
echo "========================================="

# If coverage was generated, show the location
if [ -d "htmlcov" ]; then
    echo -e "\n${YELLOW}Coverage report available at: htmlcov/index.html${NC}"
fi