# ---- app/db.py ----
import chromadb
from app import config

client = chromadb.PersistentClient(config.CHROMA_DIR)
collection = client.get_or_create_collection(name="bbc_chunks")