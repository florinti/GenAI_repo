# ---- app/models.py ----
from pydantic import BaseModel
from typing import List, Dict, Any

class QueryRequest(BaseModel):
    query: str
    top_k: int = 10
    min_score: float = 0.3

class RerankedChunk(BaseModel):
    text: str
    score: float
    source_url: str
    depth: int

class QueryResponse(BaseModel):
    answer: str
    relevant_chunks: List[RerankedChunk]
    used_prompt: Dict[str, Any]
    short_term_memory: str
    long_term_memory: Dict[str, Any]