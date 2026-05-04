"""Test search functionality."""
from whoosh.index import open_dir
from whoosh.qparser import QueryParser

idx = open_dir('whoosh_index')
with idx.searcher() as searcher:
    # Try searching for common words
    parser = QueryParser("content", idx.schema)

    test_queries = ["application", "install", "studio", "modeling", "*"]

    for query_str in test_queries:
        query = parser.parse(query_str)
        results = searcher.search(query, limit=5)
        print(f"\nQuery: '{query_str}'")
        print(f"Results found: {len(results)}")

        for hit in results:
            print(f"  - Doc: {hit['document_id']}, Page: {hit['page_number']}")
            content = hit.get('content', '')
            print(f"    Content length: {len(content)} chars")
            if len(content) > 0:
                print(f"    Preview: {content[:150]}...")
            else:
                print(f"    Content is EMPTY!")
