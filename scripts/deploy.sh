#!/bin/bash
# Complete deployment script for Document Search & Retrieval System
# This script sets up the entire application from scratch

set -e

echo "🚀 Document Search & Retrieval System - Deployment Script"
echo "=========================================================="
echo ""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}ℹ️  $1${NC}"
}

# Check prerequisites
echo "1️⃣  Checking prerequisites..."
echo ""

# Check Python version
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | awk '{print $2}')
print_success "Python $PYTHON_VERSION found"

# Check Docker
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed"
    exit 1
fi

if ! docker info > /dev/null 2>&1; then
    print_error "Docker is not running. Please start Docker."
    exit 1
fi
print_success "Docker is running"

# Check docker-compose
if ! command -v docker-compose &> /dev/null; then
    print_error "docker-compose is not installed"
    exit 1
fi
print_success "docker-compose found"

echo ""
echo "2️⃣  Setting up Python virtual environment..."
echo ""

# Create virtual environment if it doesn't exist
if [ ! -d ".myenv" ]; then
    python3 -m venv .myenv
    print_success "Virtual environment created"
else
    print_info "Virtual environment already exists"
fi

# Activate virtual environment
source .myenv/bin/activate
print_success "Virtual environment activated"

echo ""
echo "3️⃣  Installing Python dependencies..."
echo ""

pip install -q --upgrade pip
pip install -q -r requirements.txt
print_success "Dependencies installed"

echo ""
echo "4️⃣  Checking environment configuration..."
echo ""

# Check if .env file exists
if [ ! -f ".env" ]; then
    print_error ".env file not found"
    echo ""
    echo "Please create a .env file with the following variables:"
    echo "  VISION_AGENT_API_KEY=your_landingai_key"
    echo "  ANTHROPIC_API_KEY=your_anthropic_key"
    echo "  ELASTICSEARCH_URL=http://localhost:9200"
    echo "  DATABASE_URL=postgresql://postgres:postgres@localhost:5432/docsearch"
    echo "  API_KEY=your_secure_api_key"
    echo "  PDF_STORAGE_PATH=./data/pdfs"
    echo ""
    echo "See README.md for complete .env configuration"
    exit 1
fi
print_success ".env file found"

# Check required environment variables
source .env
REQUIRED_VARS=("VISION_AGENT_API_KEY" "ANTHROPIC_API_KEY" "API_KEY" "DATABASE_URL")
for var in "${REQUIRED_VARS[@]}"; do
    if [ -z "${!var}" ]; then
        print_error "Environment variable $var is not set in .env"
        exit 1
    fi
done
print_success "Required environment variables are set"

echo ""
echo "5️⃣  Starting Docker services..."
echo ""

# Start services
docker-compose up -d
print_success "Docker services started"

# Wait for Elasticsearch
print_info "Waiting for Elasticsearch to be ready..."
until curl -s http://localhost:9200/_cluster/health > /dev/null 2>&1; do
    echo -n "."
    sleep 2
done
echo ""
print_success "Elasticsearch is ready"

# Wait for PostgreSQL
print_info "Waiting for PostgreSQL to be ready..."
until docker-compose exec -T postgres pg_isready -U postgres > /dev/null 2>&1; do
    echo -n "."
    sleep 1
done
echo ""
print_success "PostgreSQL is ready"

echo ""
echo "6️⃣  Initializing database..."
echo ""

# Run database initialization
python3 scripts/init_database.py
print_success "Database initialized"

echo ""
echo "7️⃣  Creating Elasticsearch index..."
echo ""

# Run Elasticsearch initialization
python3 scripts/init_elasticsearch.py
print_success "Elasticsearch index created"

echo ""
echo "8️⃣  Creating data directories..."
echo ""

# Create PDF storage directory
mkdir -p data/pdfs
print_success "Data directories created"

echo ""
echo "✨ Deployment complete!"
echo ""
echo "=========================================================="
echo "📊 Service Status:"
echo "  - Elasticsearch: http://localhost:9200"
echo "  - PostgreSQL:    localhost:5432"
echo ""
echo "🚀 To start the application:"
echo "  uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload"
echo ""
echo "🌐 Application will be available at:"
echo "  - Search UI:  http://localhost:8000/"
echo "  - API Docs:   http://localhost:8000/docs"
echo "  - Health:     http://localhost:8000/health"
echo ""
echo "🧪 To run tests:"
echo "  pytest"
echo ""
echo "📋 Useful commands:"
echo "  - View logs:       docker-compose logs -f"
echo "  - Stop services:   docker-compose down"
echo "  - Restart services: docker-compose restart"
echo "=========================================================="
echo ""
