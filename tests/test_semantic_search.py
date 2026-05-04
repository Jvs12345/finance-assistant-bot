import sys
import os
from pprint import pprint

# Ensure src is in path
sys.path.append(os.getcwd())

from src.db.elasticsearch_client import get_elasticsearch_client
from src.services.embedding_service import EmbeddingService, EmbeddingProvider

def test_semantic_search():
    print("Initializing Elasticsearch Client...")
    client = get_elasticsearch_client()
    
    query = "configure BMS" 
    print(f"\nTesting query: '{query}'")
    
    try:
        # 1. Standard search (hybrid if implemented)
        results = client.search(query, limit=3)
        
        print(f"\nFound {len(results)} results:")
        for i, r in enumerate(results, 1):
            print(f"\n{i}. {r.get('filename')} (Score: {r.get('score')})")
            content = r.get('content', '')[:200].replace('\n', ' ')
            print(f"   Content: {content}...")
            if 'embedding' in r:
                print("   [Has Embedding]")
            else:
                print("   [No Embedding field in result]")
                
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_semantic_search()
