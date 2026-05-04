"""
Semantic search engine using BM25 + optional embeddings + LLM reranking.
Provides Google-like search behavior over PDF sections.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import json
import re

from rank_bm25 import BM25Okapi
import numpy as np

from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SearchResult:
    """A single search result with relevance score."""
    doc_id: str
    title: str
    snippet: str
    score: float
    source: str
    page_num: Optional[int] = None
    section_type: Optional[str] = None


@dataclass
class SearchResponse:
    """Complete search response with direct answer and ranked results."""
    query: str
    direct_answer: Optional[str]
    results: List[SearchResult]
    total_results: int
    search_time_ms: float


class SemanticSearchEngine:
    """
    Semantic search engine that provides Google-like search over PDF sections.

    Architecture:
    1. BM25 for initial retrieval (keyword + statistical relevance)
    2. Optional: Embedding-based semantic similarity
    3. Optional: LLM-based reranking and answer generation
    """

    def __init__(
        self,
        index_path: Path,
        use_embeddings: bool = False,
        use_llm_rerank: bool = False
    ):
        """
        Initialize the semantic search engine.

        Args:
            index_path: Path to the indexed documents directory
            use_embeddings: Whether to use embedding-based similarity
            use_llm_rerank: Whether to use LLM for reranking and answer generation
        """
        self.index_path = Path(index_path)
        self.use_embeddings = use_embeddings
        self.use_llm_rerank = use_llm_rerank

        # Storage for indexed documents
        self.documents: List[Dict[str, Any]] = []
        self.doc_texts: List[str] = []
        self.bm25: Optional[BM25Okapi] = None

        # Load index
        self._load_index()

    def _load_index(self) -> None:
        """Load the document index from disk."""
        index_file = self.index_path / "index.json"

        if not index_file.exists():
            logger.warning(f"No index found at {index_file}")
            return

        try:
            with open(index_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.documents = data.get('documents', [])

            # Extract text for BM25
            self.doc_texts = [doc.get('text', '') for doc in self.documents]

            # Build BM25 index
            tokenized_docs = [self._tokenize(text) for text in self.doc_texts]
            self.bm25 = BM25Okapi(tokenized_docs)

            logger.info(f"Loaded {len(self.documents)} documents from index")

        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            raise

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text for BM25.

        Args:
            text: Input text

        Returns:
            List of tokens
        """
        # Simple tokenization: lowercase, split on non-alphanumeric
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        return tokens

    def _bm25_search(self, query: str, top_k: int = 20) -> List[Tuple[int, float]]:
        """
        Perform BM25 search.

        Args:
            query: Search query
            top_k: Number of top results to return

        Returns:
            List of (doc_index, score) tuples
        """
        if self.bm25 is None:
            return []

        # Tokenize query
        query_tokens = self._tokenize(query)

        # Get BM25 scores
        scores = self.bm25.get_scores(query_tokens)

        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]

        # Return (index, score) pairs
        results = [(int(idx), float(scores[idx])) for idx in top_indices if scores[idx] > 0]

        return results

    def _generate_snippet(self, text: str, query: str, max_length: int = 200) -> str:
        """
        Generate a snippet highlighting relevant parts of the text.

        Args:
            text: Full text
            query: Search query
            max_length: Maximum snippet length in characters

        Returns:
            Snippet with query context
        """
        # Find query terms in text
        query_tokens = self._tokenize(query)
        text_lower = text.lower()

        # Find first occurrence of any query term
        best_pos = -1
        for token in query_tokens:
            pos = text_lower.find(token)
            if pos != -1 and (best_pos == -1 or pos < best_pos):
                best_pos = pos

        if best_pos == -1:
            # No match found, return beginning
            return text[:max_length] + ("..." if len(text) > max_length else "")

        # Extract context around match
        start = max(0, best_pos - max_length // 2)
        end = min(len(text), start + max_length)

        snippet = text[start:end]

        # Add ellipsis if truncated
        if start > 0:
            snippet = "..." + snippet
        if end < len(text):
            snippet = snippet + "..."

        return snippet.strip()

    def _create_search_results(
        self,
        query: str,
        bm25_results: List[Tuple[int, float]]
    ) -> List[SearchResult]:
        """
        Convert BM25 results to SearchResult objects.

        Args:
            query: Search query
            bm25_results: List of (doc_index, score) tuples

        Returns:
            List of SearchResult objects
        """
        results = []

        for idx, score in bm25_results:
            if idx >= len(self.documents):
                continue

            doc = self.documents[idx]

            # Create search result
            result = SearchResult(
                doc_id=doc.get('id', f'doc_{idx}'),
                title=doc.get('title', 'Untitled Section'),
                snippet=self._generate_snippet(doc.get('text', ''), query),
                score=score,
                source=doc.get('source', 'Unknown'),
                page_num=doc.get('page_num'),
                section_type=doc.get('section_type')
            )

            results.append(result)

        return results

    def search(
        self,
        query: str,
        top_k: int = 8,
        generate_answer: bool = True
    ) -> SearchResponse:
        """
        Perform semantic search.

        Args:
            query: Search query
            top_k: Number of results to return
            generate_answer: Whether to generate a direct answer

        Returns:
            SearchResponse with results and optional direct answer
        """
        import time
        start_time = time.time()

        # Step 1: BM25 search
        bm25_results = self._bm25_search(query, top_k=top_k * 2)

        # Step 2: Create search results
        results = self._create_search_results(query, bm25_results)[:top_k]

        # Step 3: Generate direct answer (if requested)
        direct_answer = None
        if generate_answer and results:
            direct_answer = self._generate_direct_answer(query, results)

        # Calculate search time
        search_time = (time.time() - start_time) * 1000

        return SearchResponse(
            query=query,
            direct_answer=direct_answer,
            results=results,
            total_results=len(results),
            search_time_ms=search_time
        )

    def _generate_direct_answer(
        self,
        query: str,
        results: List[SearchResult]
    ) -> Optional[str]:
        """
        Generate a direct answer from top results.

        Args:
            query: Search query
            results: Top search results

        Returns:
            Direct answer string or None
        """
        if not results:
            return None

        # Simple heuristic: use top result's snippet as direct answer
        # In a full implementation, this would use an LLM
        top_result = results[0]

        # Only provide direct answer if score is high enough
        if top_result.score > 5.0:  # BM25 threshold
            return top_result.snippet

        return None

    def format_search_response_text(self, response: SearchResponse) -> str:
        """
        Format search response as Google-like text output.

        Args:
            response: SearchResponse object

        Returns:
            Formatted text string
        """
        output = []

        # Direct answer
        if response.direct_answer:
            output.append("=== DIRECT ANSWER ===")
            output.append(response.direct_answer)
            output.append("")

        # Search results
        output.append(f"=== SEARCH RESULTS ({response.total_results} results in {response.search_time_ms:.0f}ms) ===")
        output.append("")

        for i, result in enumerate(response.results, 1):
            # Title
            output.append(f"{i}. {result.title}")

            # Source info
            source_info = f"   Source: {result.source}"
            if result.page_num is not None:
                source_info += f", Page {result.page_num}"
            if result.section_type:
                source_info += f" ({result.section_type})"
            output.append(source_info)

            # Snippet
            output.append(f"   {result.snippet}")

            # Score (for debugging)
            output.append(f"   [Relevance: {result.score:.2f}]")
            output.append("")

        return "\n".join(output)
