#!/bin/bash

# PDF Question Extractor API Test Script
# This script demonstrates how to test all the implemented API endpoints

BASE_URL="http://localhost:8000"
API_BASE="${BASE_URL}/api"

echo "=== PDF Question Extractor API Test Script ==="
echo "Base URL: $BASE_URL"
echo "API Base: $API_BASE"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to make API calls and display results
test_endpoint() {
    local method=$1
    local endpoint=$2
    local data=$3
    local description=$4
    
    echo -e "${BLUE}Testing: $description${NC}"
    echo -e "${YELLOW}$method $endpoint${NC}"
    
    if [ -n "$data" ]; then
        echo -e "${YELLOW}Data: $data${NC}"
        response=$(curl -s -X $method "$endpoint" \
            -H "Content-Type: application/json" \
            -d "$data" \
            -w "\nHTTP_STATUS:%{http_code}")
    else
        response=$(curl -s -X $method "$endpoint" \
            -w "\nHTTP_STATUS:%{http_code}")
    fi
    
    # Extract HTTP status and body
    http_status=$(echo "$response" | grep "HTTP_STATUS:" | cut -d: -f2)
    body=$(echo "$response" | sed '/HTTP_STATUS:/d')
    
    if [ "$http_status" -ge 200 ] && [ "$http_status" -lt 300 ]; then
        echo -e "${GREEN}✓ Success (HTTP $http_status)${NC}"
    else
        echo -e "${RED}✗ Failed (HTTP $http_status)${NC}"
    fi
    
    echo "Response:"
    echo "$body" | python3 -m json.tool 2>/dev/null || echo "$body"
    echo ""
    echo "---"
    echo ""
}

# 1. Health Check
test_endpoint "GET" "$BASE_URL/health" "" "Health Check"

# 2. Upload PDF (requires a test PDF file)
# Note: This requires multipart/form-data, so we'll show the curl command instead
echo -e "${BLUE}Testing: Upload PDF Files${NC}"
echo -e "${YELLOW}POST $API_BASE/upload${NC}"
echo "Note: This endpoint requires multipart/form-data. Example curl command:"
echo ""
cat << 'EOF'
curl -X POST "$API_BASE/upload" \
  -F "files=@/path/to/your/test.pdf" \
  -F "files=@/path/to/your/test2.pdf" \
  -F "request_data={\"store_to_db\": true, \"generate_embeddings\": true, \"max_concurrent\": 2}"
EOF
echo ""
echo "---"
echo ""

# 3. Get Processing Status
test_endpoint "GET" "$API_BASE/process/test-processing-id" "" "Get Processing Status"

# 4. Get Questions (Extracted Table)
test_endpoint "GET" "$API_BASE/questions?page=1&per_page=10&table=extracted" "" "Get Extracted Questions (Page 1)"

# 5. Get Questions (Permanent Table)
test_endpoint "GET" "$API_BASE/questions?page=1&per_page=10&table=permanent" "" "Get Permanent Questions (Page 1)"

# 6. Get Questions with Filters
test_endpoint "GET" "$API_BASE/questions?status_filter=approved&year_filter=2023&per_page=5" "" "Get Approved Questions from 2023"

# 7. Search Questions
test_endpoint "GET" "$API_BASE/questions?search=mathematics&per_page=5" "" "Search Questions containing 'mathematics'"

# 8. Update Question (requires existing question ID - using ID 1 as example)
update_data='{
  "question_text": "Updated question text for testing",
  "marks": 10,
  "topics": ["Mathematics", "Algebra"],
  "status": "approved"
}'
test_endpoint "PUT" "$API_BASE/questions/1?table=extracted" "$update_data" "Update Question ID 1"

# 9. Bulk Operations
bulk_data='{
  "question_ids": [1, 2, 3],
  "operation": "approve"
}'
test_endpoint "POST" "$API_BASE/questions/bulk?table=extracted" "$bulk_data" "Bulk Approve Questions"

# 10. Bulk Status Update
bulk_status_data='{
  "question_ids": [1, 2],
  "operation": "update_status",
  "new_status": "approved"
}'
test_endpoint "POST" "$API_BASE/questions/bulk?table=extracted" "$bulk_status_data" "Bulk Update Status"

# 11. Save Approved Questions
save_data='{
  "question_ids": null,
  "clear_extracted": false
}'
test_endpoint "POST" "$API_BASE/questions/save" "$save_data" "Save All Approved Questions"

# 12. Export Questions (CSV)
test_endpoint "GET" "$API_BASE/export?format=csv&include_metadata=true" "" "Export Questions to CSV"

# 13. Export Questions (JSON with filters)
test_endpoint "GET" "$API_BASE/export?format=json&year_filter=2023&status_filter=approved" "" "Export 2023 Approved Questions to JSON"

# 14. Get Statistics
test_endpoint "GET" "$API_BASE/stats" "" "Get System Statistics"

# 15. Download File (example - requires actual filename)
echo -e "${BLUE}Testing: Download Exported File${NC}"
echo -e "${YELLOW}GET $API_BASE/download/{filename}${NC}"
echo "Note: This endpoint requires an actual filename from a previous export."
echo "Example:"
echo 'curl -O "$API_BASE/download/questions_export_20241204_123456.csv"'
echo ""
echo "---"
echo ""

# WebSocket Test
echo -e "${BLUE}Testing: WebSocket Connection${NC}"
echo -e "${YELLOW}WebSocket: ws://localhost:8000/api/ws/processing${NC}"
echo "Note: WebSocket testing requires a WebSocket client. Example using wscat:"
echo ""
cat << 'EOF'
# Install wscat: npm install -g wscat
# Connect to WebSocket
wscat -c ws://localhost:8000/api/ws/processing

# Send messages:
{"type": "ping"}
{"type": "start_processing", "data": {"file_path": "/path/to/file.pdf"}}
EOF
echo ""
echo "---"
echo ""

echo -e "${GREEN}=== API Test Script Complete ===${NC}"
echo ""
echo "Additional Notes:"
echo "- Ensure the FastAPI server is running: python app.py"
echo "- Some endpoints require existing data in the database"
echo "- File upload and download endpoints require actual files"
echo "- WebSocket endpoint provides real-time updates during processing"
echo ""
echo "To run the server:"
echo "cd /path/to/pdf_question_extractor"
echo "python app.py"
echo ""
echo "Or with uvicorn directly:"
echo "uvicorn app:app --host 0.0.0.0 --port 8000 --reload"