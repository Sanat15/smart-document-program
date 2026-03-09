"""Search endpoint for document queries with stage-wise latency tracking."""
from fastapi import APIRouter, HTTPException
from typing import Optional, Dict
import time
import logging
from dataclasses import dataclass, asdict

from app.models import SearchRequest, SearchResponse, SearchResult
from app.embedding_service import get_embedding_service
from app.vector_store import get_vector_store
from app.cache import get_cache_service
from app.config import get_settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["search"])

settings = get_settings()


@dataclass
class LatencyBreakdown:
    """Stage-wise latency breakdown for a search query."""
    cache_lookup_ms: float = 0.0
    embedding_ms: float = 0.0
    retrieval_ms: float = 0.0
    reranking_ms: float = 0.0
    formatting_ms: float = 0.0
    total_ms: float = 0.0
    embedding_cached: bool = False
    result_cached: bool = False
    
    def to_dict(self) -> Dict:
        return asdict(self)


@router.post("/search", response_model=SearchResponse)
async def search_documents(request: SearchRequest):
    """
    Search documents with natural language query.
    
    - Generates embedding for query (with caching)
    - Performs vector similarity search
    - Returns ranked results with metadata
    - Caches results for repeated queries
    - Provides stage-wise latency breakdown
    """
    start_time = time.time()
    latency = LatencyBreakdown()
    
    query = request.query.strip()
    top_k = request.top_k
    # Fix: Use 'is not None' to allow 0.0 as a valid threshold
    threshold = request.score_threshold if request.score_threshold is not None else settings.similarity_threshold
    
    if not query:
        raise HTTPException(
            status_code=400,
            detail="Query cannot be empty"
        )
    
    # Check result cache first
    cache_start = time.time()
    cache = get_cache_service()
    cached_results = cache.get(query, top_k, threshold)
    latency.cache_lookup_ms = (time.time() - cache_start) * 1000
    
    if cached_results is not None:
        latency.total_ms = (time.time() - start_time) * 1000
        latency.result_cached = True
        cache.record_latency(latency.total_ms, cached=True)
        
        logger.info(f"Cache HIT for query: '{query[:50]}...' ({latency.total_ms:.1f}ms)")
        
        return SearchResponse(
            query=query,
            results=[SearchResult(**r) for r in cached_results],
            total_results=len(cached_results),
            query_time_ms=round(latency.total_ms, 2),
            cached=True,
            latency_breakdown=latency.to_dict()
        )
    
    try:
        # Check embedding cache
        embed_start = time.time()
        cached_embedding, embedding_cached = cache.get_embedding(query)
        
        if cached_embedding is not None:
            query_embedding = cached_embedding
            latency.embedding_cached = True
        else:
            # Generate query embedding
            embed_service = get_embedding_service()
            query_embedding = embed_service.encode_text(query)
            # Cache the embedding
            cache.set_embedding(query, query_embedding)
        
        latency.embedding_ms = (time.time() - embed_start) * 1000
        
        # Search vector store
        retrieval_start = time.time()
        vector_store = get_vector_store()
        results = vector_store.search(
            query_vector=query_embedding,
            top_k=top_k,
            score_threshold=threshold
        )
        latency.retrieval_ms = (time.time() - retrieval_start) * 1000
        
        # Optional re-ranking
        if request.use_reranking and results:
            try:
                from app.reranker import get_reranker_service
                rerank_start = time.time()
                reranker = get_reranker_service(enabled=True)
                
                # Convert to format for reranker
                results_for_rerank = [
                    {'text': r['text'], 'score': r['score'], 'metadata': r['metadata']}
                    for r in results
                ]
                
                reranked_results, rerank_time = reranker.rerank(query, results_for_rerank, top_k=top_k)
                
                # Update results with reranked order
                results = [
                    {'text': r['text'], 'score': r.get('similarity_score', r['score']), 'metadata': r['metadata']}
                    for r in reranked_results
                ]
                latency.reranking_ms = (time.time() - rerank_start) * 1000
            except Exception as e:
                logger.warning(f"Re-ranking failed, using original results: {e}")
                latency.reranking_ms = 0.0
        
        # Format results
        format_start = time.time()
        formatted_results = []
        for r in results:
            formatted_results.append(SearchResult(
                text=r['text'],
                pdf_name=r['metadata']['pdf_name'],
                page_number=r['metadata']['page'],
                chunk_index=r['metadata']['chunk_index'],
                similarity_score=round(r['score'], 4)
            ))
        latency.formatting_ms = (time.time() - format_start) * 1000
        
        # Cache results
        cache.set(
            query, 
            top_k, 
            [r.model_dump() for r in formatted_results],
            threshold
        )
        
        latency.total_ms = (time.time() - start_time) * 1000
        cache.record_latency(latency.total_ms, cached=False)
        
        logger.info(
            f"Search: '{query[:50]}...' -> {len(formatted_results)} results "
            f"({latency.total_ms:.1f}ms, emb: {latency.embedding_ms:.1f}ms, "
            f"ret: {latency.retrieval_ms:.1f}ms)"
        )
        
        return SearchResponse(
            query=query,
            results=formatted_results,
            total_results=len(formatted_results),
            query_time_ms=round(latency.total_ms, 2),
            cached=False,
            latency_breakdown=latency.to_dict()
        )
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Search error: {str(e)}"
        )


@router.get("/search/stats")
async def get_search_stats():
    """Get search and cache statistics."""
    try:
        cache = get_cache_service()
        vector_store = get_vector_store()
        
        cache_stats = cache.get_stats()
        collection_info = vector_store.get_collection_info()
        
        return {
            "cache": cache_stats,
            "vector_store": collection_info
        }
        
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting stats: {str(e)}"
        )
