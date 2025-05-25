# ---- main.py ----
from fastapi import FastAPI
from app.routes import router
from app.scraping import scrape_url
from app.db import collection
from app.llm import embedder
from app import config
import asyncio

app = FastAPI()
app.include_router(router)

@app.on_event("startup")
async def startup_event():
    print("Starting scrape and index...")
    existing_ids = collection.get().get('ids', [])
    if existing_ids:
        collection.delete(ids=existing_ids)
    raw_chunks = await scrape_url(config.TARGET_URL, config.DEPTH, set())
    ids, embeddings, metadatas = [], [], []
    for idx, (text, src, lvl) in enumerate(raw_chunks):
        emb = embedder.encode(text).tolist()
        ids.append(f"chunk_{lvl}_{idx}")
        embeddings.append(emb)
        metadatas.append({"text": text, "source_url": src, "depth": lvl})
    collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas)
    print(f"Indexed {len(ids)} chunks into ChromaDB.")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
