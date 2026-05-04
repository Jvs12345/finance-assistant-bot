"""
Test script for the semantic PDF search system.
Creates sample data and tests all components.
"""

from pathlib import Path
import json

from src.services.pdf_section_extractor import PDFSection
from src.services.semantic_indexer import SemanticIndexer
from src.services.semantic_search_engine import SemanticSearchEngine


def create_test_sections() -> list:
    """Create sample sections for testing."""
    sections = [
        PDFSection(
            id="test001",
            text="The installation process requires Python 3.8 or higher. First, download the installer from the official website. Then, run the installation wizard and follow the on-screen instructions.",
            page_num=1,
            section_type="paragraph",
            title="Installation Requirements",
            source_file="manual.pdf",
            confidence=1.0
        ),
        PDFSection(
            id="test002",
            text="Configuration Guide",
            page_num=2,
            section_type="heading",
            title="Configuration Guide",
            source_file="manual.pdf",
            confidence=1.0
        ),
        PDFSection(
            id="test003",
            text="To configure the system, navigate to Settings > Configuration. Here you can adjust parameters such as timeout values, connection pools, and logging levels. Save your changes and restart the service.",
            page_num=2,
            section_type="paragraph",
            title="System Configuration Steps",
            source_file="manual.pdf",
            confidence=1.0
        ),
        PDFSection(
            id="test004",
            text="Troubleshooting Common Issues",
            page_num=3,
            section_type="heading",
            title="Troubleshooting Common Issues",
            source_file="manual.pdf",
            confidence=1.0
        ),
        PDFSection(
            id="test005",
            text="If the application fails to start, check the log files in the logs directory. Common causes include port conflicts, missing dependencies, or incorrect configuration values. Refer to the error codes table for specific solutions.",
            page_num=3,
            section_type="paragraph",
            title="Application Startup Failures",
            source_file="manual.pdf",
            confidence=1.0
        ),
    ]
    return sections


def test_indexing():
    """Test the indexing process."""
    print("\n=== Testing Indexing ===")

    # Create test index directory
    test_index_dir = Path("./data/test_semantic_index")
    test_index_dir.mkdir(parents=True, exist_ok=True)

    # Initialize indexer
    indexer = SemanticIndexer(index_dir=test_index_dir)

    # Create and index test sections
    sections = create_test_sections()
    indexer.index_sections(sections, "manual.pdf")

    # Get summary
    summary = indexer.get_summary()
    print(f"[OK] Indexed {summary['total_sections']} sections")
    print(f"[OK] Section types: {summary['section_types']}")

    return test_index_dir


def test_search(index_dir: Path):
    """Test the search functionality."""
    print("\n=== Testing Search ===")

    # Initialize search engine
    engine = SemanticSearchEngine(index_path=index_dir)

    # Test queries
    queries = [
        "how to install",
        "configuration settings",
        "troubleshooting startup problems",
        "python version requirements"
    ]

    for query in queries:
        print(f"\nQuery: '{query}'")
        response = engine.search(query, top_k=3, generate_answer=True)

        print(f"  Found {response.total_results} results in {response.search_time_ms:.1f}ms")

        if response.direct_answer:
            print(f"  Direct answer: {response.direct_answer[:80]}...")

        if response.results:
            top_result = response.results[0]
            print(f"  Top result: {top_result.title}")
            print(f"    Score: {top_result.score:.2f}")
            print(f"    Source: {top_result.source}, Page {top_result.page_num}")


def test_full_output(index_dir: Path):
    """Test full formatted output."""
    print("\n=== Testing Full Output Format ===")

    engine = SemanticSearchEngine(index_path=index_dir)

    query = "how to configure the system"
    response = engine.search(query, top_k=5, generate_answer=True)

    # Display formatted output
    output = engine.format_search_response_text(response)
    print(output)


def test_issues_logging():
    """Test issues logging."""
    print("\n=== Testing Issues Logging ===")

    test_index_dir = Path("./data/test_semantic_index")
    indexer = SemanticIndexer(index_dir=test_index_dir)

    # Create section with low confidence
    low_conf_section = PDFSection(
        id="test_low",
        text="This text was extracted with low OCR confidence",
        page_num=10,
        section_type="paragraph",
        title="Low Confidence Section",
        source_file="scanned.pdf",
        confidence=0.65
    )

    indexer.index_sections([low_conf_section], "scanned.pdf")

    # Check issues
    issues = indexer.get_issues()
    print(f"[OK] Logged {len(issues)} issues")

    if issues:
        for issue in issues[-3:]:  # Show last 3 issues
            print(f"  [{issue['type']}] {issue['message']}")


def main():
    """Run all tests."""
    print("=" * 60)
    print("  Semantic PDF Search System - Test Suite")
    print("=" * 60)

    try:
        # Test indexing
        index_dir = test_indexing()

        # Test search
        test_search(index_dir)

        # Test full output
        test_full_output(index_dir)

        # Test issues
        test_issues_logging()

        print("\n" + "=" * 60)
        print("  [SUCCESS] All Tests Passed")
        print("=" * 60)

    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
