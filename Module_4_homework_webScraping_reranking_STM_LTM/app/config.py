
# ---- app/config.py ----
TARGET_URL = "https://www.bbc.com/news"
BASE_DOMAIN = "https://www.bbc.com"
DEPTH = 2
CHROMA_DIR = "./chroma_db"
HISTORY_FILE = "history.json"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
RERANK_MODEL_NAME = "cross-encoder/ms-marco-TinyBERT-L-6"
LLAMA_MODEL = "llama2"
LLAMA_HOST = "http://localhost"
LLAMA_PORT = 11434
TOPIC_SIM_THRESHOLD = 0.5