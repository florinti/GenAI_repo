# ---- app/llm.py ----
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity
import ollama
from app import config

embedder = SentenceTransformer(config.EMBED_MODEL_NAME)
reranker = CrossEncoder(config.RERANK_MODEL_NAME)
tool = ollama.Client(host=f"{config.LLAMA_HOST}:{config.LLAMA_PORT}")