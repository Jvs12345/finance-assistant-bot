#!/bin/bash
# Start Docker services for Document Search & Retrieval System

set -e

echo "🚀 Starting Document Search services..."
echo ""

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Error: Docker is not running"
    echo "Please start Docker and try again"
    exit 1
fi

# Start services
echo "📦 Starting Elasticsearch and PostgreSQL..."
docker-compose up -d

echo ""
echo "⏳ Waiting for services to be healthy..."
echo ""

# Wait for Elasticsearch
echo "Waiting for Elasticsearch..."
until curl -s http://localhost:9200/_cluster/health > /dev/null 2>&1; do
    echo -n "."
    sleep 2
done
echo " ✅ Elasticsearch is ready"

# Wait for PostgreSQL
echo "Waiting for PostgreSQL..."
until docker-compose exec -T postgres pg_isready -U docsearch > /dev/null 2>&1; do
    echo -n "."
    sleep 1
done
echo " ✅ PostgreSQL is ready"

echo ""
echo "✨ All services are running!"
echo ""
echo "📊 Service URLs:"
echo "  - Elasticsearch: http://localhost:9200"
echo "  - PostgreSQL:    localhost:5432"
echo ""
echo "🔍 Check service status: docker-compose ps"
echo "📋 View logs:           docker-compose logs -f"
echo "🛑 Stop services:       docker-compose down"
echo ""
