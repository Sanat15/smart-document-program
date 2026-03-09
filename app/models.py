"""Pydantic models for request/response schemas."""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from uuid import UUID


# ============== Request Models ==============

class SearchRequest(BaseModel):
    """Request model for document search."""
    query: str = Field(..., min_length=1, max_length=1000, description="Search query text")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of results to return")
    score_threshold: Optional[float] = Field(default=0.5, ge=0, le=1, description="Minimum similarity score")
    use_reranking: bool = Field(default=False, description="Enable cross-encoder re-ranking for improved accuracy")


class UploadResponse(BaseModel):
    """Response model for PDF upload."""
    status: str
    filename: str
    file_id: str
    chunks_created: int
    total_pages: int
    processing_time_seconds: float


# ============== Response Models ==============

class ChunkMetadata(BaseModel):
    """Metadata for a document chunk."""
    pdf_name: str
    page: int
    chunk_index: int
    token_count: int


class SearchResult(BaseModel):
    """Single search result."""
    text: str
    pdf_name: str
    page_number: int
    chunk_index: int
    similarity_score: float


class SearchResponse(BaseModel):
    """Response model for search endpoint."""
    query: str
    results: List[SearchResult]
    total_results: int
    query_time_ms: float
    cached: bool = False
    latency_breakdown: Optional[Dict[str, Any]] = Field(
        default=None, 
        description="Stage-wise latency breakdown (embedding, retrieval, formatting)"
    )


# ============== Database Models (for reference) ==============

class PDFDocument(BaseModel):
    """PDF document metadata."""
    id: str
    filename: str
    file_path: str
    total_pages: int
    file_size_mb: float
    uploaded_at: datetime
    status: str
    chunk_count: int


class DocumentChunk(BaseModel):
    """Document chunk with metadata."""
    id: str
    pdf_id: str
    page_number: int
    chunk_index: int
    content: str
    token_count: int
    created_at: datetime


# ============== Health Check ==============

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    services: Dict[str, str]


class MetricsResponse(BaseModel):
    """System metrics response."""
    total_documents: int
    total_chunks: int
    cache_hit_rate: float
    avg_query_latency_ms: float
