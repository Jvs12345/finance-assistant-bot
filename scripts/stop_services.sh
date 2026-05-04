#!/bin/bash
# Stop all services for Document Search & Retrieval System

set -e

echo "🛑 Stopping Document Search services..."
echo ""

# Stop Docker services
docker-compose down

echo ""
echo "✅ All services stopped"
echo ""
echo "To restart services, run:"
echo "  ./scripts/start_services.sh"
echo "or"
echo "  docker-compose up -d"
echo ""
