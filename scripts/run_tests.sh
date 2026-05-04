#!/bin/bash
# Run tests for Document Search & Retrieval System

set -e

echo "🧪 Running Document Search tests..."
echo ""

# Check if virtual environment exists
if [ ! -d ".myenv" ]; then
    echo "❌ Virtual environment not found. Run scripts/deploy.sh first."
    exit 1
fi

# Activate virtual environment
source .myenv/bin/activate

# Run tests based on arguments
if [ "$1" == "coverage" ]; then
    echo "📊 Running tests with coverage report..."
    pytest --cov=src --cov-report=html --cov-report=term
    echo ""
    echo "📈 Coverage report generated in htmlcov/index.html"
elif [ "$1" == "unit" ]; then
    echo "🔬 Running unit tests only..."
    pytest -m unit -v
elif [ "$1" == "integration" ]; then
    echo "🔗 Running integration tests only..."
    pytest -m integration -v
elif [ "$1" == "e2e" ]; then
    echo "🌐 Running end-to-end tests only..."
    pytest -m e2e -v
elif [ "$1" == "fast" ]; then
    echo "⚡ Running fast tests (unit + integration)..."
    pytest -v -m "not e2e"
else
    echo "🧪 Running all tests..."
    pytest -v
fi

echo ""
echo "✅ Tests complete!"
echo ""
echo "Available test modes:"
echo "  ./scripts/run_tests.sh           # Run all tests"
echo "  ./scripts/run_tests.sh coverage  # Run with coverage report"
echo "  ./scripts/run_tests.sh unit      # Unit tests only"
echo "  ./scripts/run_tests.sh integration # Integration tests only"
echo "  ./scripts/run_tests.sh e2e       # End-to-end tests only"
echo "  ./scripts/run_tests.sh fast      # Skip slow e2e tests"
echo ""
