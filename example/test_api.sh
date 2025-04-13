#!/bin/bash

# API Configuration
BASE_URL="http://localhost:8000"
USERNAME="test"
PASSWORD="test"
PROJECT="test_project"

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check for required commands
if ! command_exists curl; then
    echo -e "${RED}Error: curl is required but not installed${NC}"
    exit 1
fi

if ! command_exists jq; then
    echo -e "${RED}Error: jq is required but not installed${NC}"
    echo "You can install it with: brew install jq"
    exit 1
fi

echo -e "${BLUE}Logging in...${NC}"
# Login and get token
RESPONSE=$(curl -s -X POST "$BASE_URL/token" \
    -H "Content-Type: application/x-www-form-urlencoded" \
    -d "username=$USERNAME&password=$PASSWORD")

# Check if the response is valid JSON
if ! echo "$RESPONSE" | jq . >/dev/null 2>&1; then
    echo -e "${RED}Error: Invalid response from server${NC}"
    echo "Response: $RESPONSE"
    exit 1
fi

TOKEN=$(echo "$RESPONSE" | jq -r '.access_token')

if [ -z "$TOKEN" ] || [ "$TOKEN" = "null" ]; then
    echo -e "${RED}Login failed${NC}"
    echo "Response: $RESPONSE"
    exit 1
fi
echo -e "${GREEN}Login successful!${NC}"

# Upload files
echo -e "\n${BLUE}Uploading files...${NC}"
for file in quantum_computing.txt artificial_intelligence.txt; do
    if [ ! -f "$file" ]; then
        echo -e "${RED}Error: File $file not found${NC}"
        continue
    fi
    
    echo "Uploading $file..."
    RESPONSE=$(curl -s -X POST "$BASE_URL/api/$USERNAME/$PROJECT/files" \
        -H "Authorization: Bearer $TOKEN" \
        -F "file=@$file")
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}Error uploading $file${NC}"
        echo "Response: $RESPONSE"
    else
        echo -e "${GREEN}Uploaded $file${NC}"
    fi
done

# Make queries
echo -e "\n${BLUE}Making queries...${NC}"

# Query 1: Quantum computing basics
echo -e "\n${BLUE}Query 1: What are the key concepts in quantum computing?${NC}"
RESPONSE=$(curl -s -X POST "$BASE_URL/api/$USERNAME/$PROJECT/query" \
    -H "Authorization: Bearer $TOKEN" \
    -H "Content-Type: application/json" \
    -d '{"text": "What are the key concepts in quantum computing?"}')

if echo "$RESPONSE" | jq . >/dev/null 2>&1; then
    echo "$RESPONSE" | jq -r '.answer'
    echo -e "${GREEN}Query 1 completed${NC}"
else
    echo -e "${RED}Error in Query 1${NC}"
    echo "Response: $RESPONSE"
fi

# Query 2: AI applications
echo -e "\n${BLUE}Query 2: What are the main applications of AI?${NC}"
RESPONSE=$(curl -s -X POST "$BASE_URL/api/$USERNAME/$PROJECT/query" \
    -H "Authorization: Bearer $TOKEN" \
    -H "Content-Type: application/json" \
    -d '{"text": "What are the main applications of AI?"}')

if echo "$RESPONSE" | jq . >/dev/null 2>&1; then
    echo "$RESPONSE" | jq -r '.answer'
    echo -e "${GREEN}Query 2 completed${NC}"
else
    echo -e "${RED}Error in Query 2${NC}"
    echo "Response: $RESPONSE"
fi

# Query 3: Combined query
echo -e "\n${BLUE}Query 3: How can AI help with quantum computing?${NC}"
RESPONSE=$(curl -s -X POST "$BASE_URL/api/$USERNAME/$PROJECT/query" \
    -H "Authorization: Bearer $TOKEN" \
    -H "Content-Type: application/json" \
    -d '{"text": "How can AI help with quantum computing?"}')

if echo "$RESPONSE" | jq . >/dev/null 2>&1; then
    echo "$RESPONSE" | jq -r '.answer'
    echo -e "${GREEN}Query 3 completed${NC}"
else
    echo -e "${RED}Error in Query 3${NC}"
    echo "Response: $RESPONSE"
fi

echo -e "\n${GREEN}All operations completed!${NC}" 