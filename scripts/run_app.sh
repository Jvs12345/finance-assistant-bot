#!/bin/bash
# Start the FastAPI application
# This script ensures services are running and starts the app

set -e

echo "🚀 Starting Document Search & Retrieval System"
echo ""

# Check if virtual environment exists
if [ ! -d ".myenv" ]; then
    echo "❌ Virtual environment not found. Run scripts/deploy.sh first."
    exit 1
fi

# Activate virtual environment
source .myenv/bin/activate

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "❌ .env file not found. Please create it (see README.md)"
    exit 1
fi

# Check if services are running
echo "📊 Checking services..."

# Check Elasticsearch
if ! curl -s http://localhost:9200/_cluster/health > /dev/null 2>&1; then
    echo "⚠️  Elasticsearch is not running. Starting services..."
    docker-compose up -d

    # Wait for Elasticsearch
    echo "⏳ Waiting for Elasticsearch..."
    until curl -s http://localhost:9200/_cluster/health > /dev/null 2>&1; do
        echo -n "."
        sleep 2
    done
    echo " ✅"
else
    echo "✅ Elasticsearch is running"
fi

# Check PostgreSQL
if ! docker-compose exec -T postgres pg_isready -U postgres > /dev/null 2>&1; then
    echo "⚠️  PostgreSQL is not ready"
    docker-compose up -d postgres

    # Wait for PostgreSQL
    echo "⏳ Waiting for PostgreSQL..."
    until docker-compose exec -T postgres pg_isready -U postgres > /dev/null 2>&1; do
        echo -n "."
        sleep 1
    done
    echo " ✅"
else
    echo "✅ PostgreSQL is running"
fi

echo ""
echo "🌐 Starting FastAPI application..."
echo ""
echo "Application will be available at:"
echo "  - Search UI:  http://localhost:8000/"
echo "  - API Docs:   http://localhost:8000/docs"
echo "  - Health:     http://localhost:8000/health"
echo ""
echo "Press Ctrl+C to stop"
echo ""

# Start application with auto-reload
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
