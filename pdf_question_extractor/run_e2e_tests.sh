#!/bin/bash

# E2E Test Runner for PDF Question Extractor
# Uses Playwright for browser automation testing

echo "========================================="
echo "PDF Question Extractor - E2E Tests"
echo "========================================="

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if app is running
echo -e "${YELLOW}Checking if application is running...${NC}"
if ! curl -s http://localhost:8000/health > /dev/null; then
    echo -e "${RED}Error: Application is not running!${NC}"
    echo "Please start the application with: docker-compose up -d"
    exit 1
fi
echo -e "${GREEN}✓ Application is running${NC}"

# Install Playwright browsers if needed
echo -e "\n${YELLOW}Setting up Playwright...${NC}"
if ! command -v playwright &> /dev/null; then
    echo "Installing Playwright..."
    pip install playwright pytest-playwright
    playwright install chromium firefox webkit
fi

# Create test results directory
mkdir -p test-results/screenshots
mkdir -p test-results/videos
mkdir -p test-results/reports

# Parse command line arguments
TEST_BROWSER="${1:-all}"
HEADLESS="${2:-true}"

# Function to run tests
run_browser_tests() {
    local browser="$1"
    local test_file="${2:-tests/e2e/test_user_workflows.py}"
    
    echo -e "\n${YELLOW}Running E2E tests on ${browser}...${NC}"
    
    # Set environment variables
    export HEADLESS="$HEADLESS"
    export TEST_BASE_URL="http://localhost:8000"
    export PYTEST_CURRENT_TEST="e2e"
    
    # Run tests
    if [ "$HEADLESS" == "false" ]; then
        echo "Running in headed mode (browser visible)..."
    fi
    
    pytest "$test_file" \
        --browser="$browser" \
        --headed=$([ "$HEADLESS" == "false" ] && echo "true" || echo "false") \
        --screenshot=only-on-failure \
        --video=retain-on-failure \
        --output="test-results/reports/${browser}_report.html" \
        -v
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ ${browser} tests passed${NC}"
        return 0
    else
        echo -e "${RED}✗ ${browser} tests failed${NC}"
        return 1
    fi
}

# Main test execution
case $TEST_BROWSER in
    "chromium"|"chrome")
        run_browser_tests "chromium"
        ;;
    "firefox")
        run_browser_tests "firefox"
        ;;
    "webkit"|"safari")
        run_browser_tests "webkit"
        ;;
    "all")
        echo -e "${GREEN}Running tests on all browsers...${NC}"
        
        # Track overall status
        all_passed=true
        
        # Run on each browser
        for browser in chromium firefox webkit; do
            if ! run_browser_tests "$browser"; then
                all_passed=false
            fi
        done
        
        # Summary
        echo -e "\n========================================="
        if [ "$all_passed" = true ]; then
            echo -e "${GREEN}All E2E tests passed!${NC}"
            exit 0
        else
            echo -e "${RED}Some E2E tests failed!${NC}"
            echo -e "${YELLOW}Check test-results/ for screenshots and videos${NC}"
            exit 1
        fi
        ;;
    "performance")
        echo -e "${YELLOW}Running performance tests...${NC}"
        run_browser_tests "chromium" "tests/e2e/test_user_workflows.py::TestPerformance"
        ;;
    *)
        echo "Usage: $0 [browser] [headless]"
        echo ""
        echo "Browsers:"
        echo "  chromium/chrome - Run tests on Chromium"
        echo "  firefox         - Run tests on Firefox"  
        echo "  webkit/safari   - Run tests on WebKit (Safari)"
        echo "  all             - Run tests on all browsers (default)"
        echo "  performance     - Run performance tests only"
        echo ""
        echo "Modes:"
        echo "  true  - Run headless (default)"
        echo "  false - Run with browser visible"
        echo ""
        echo "Examples:"
        echo "  $0                    # Run all tests headless"
        echo "  $0 chromium false     # Run Chromium tests with browser visible"
        echo "  $0 performance        # Run performance tests"
        exit 1
        ;;
esac