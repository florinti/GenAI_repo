# ---- app/routes.py ----
from fastapi import APIRouter, HTTPException
from app.models import QueryRequest, QueryResponse, RerankedChunk
from app.utils import clean_text
from app.db import collection
from app.llm import embedder, reranker, tool
from app import config
from app.history import load_history, save_history
from sklearn.metrics.pairwise import cosine_similarity
import json

router = APIRouter()

@router.post("/query", response_model=QueryResponse)
async def handle_query(req: QueryRequest):
    q_emb = embedder.encode(req.query).reshape(1, -1)
    results = collection.query(
        query_embeddings=q_emb.tolist(), n_results=req.top_k * 2, include=["metadatas"]
    )
    candidates = results.get('metadatas', [[]])[0]
    if not candidates:
        raise HTTPException(status_code=404, detail="No indexed chunks available.")

    rerank_inputs = [(req.query, c['text']) for c in candidates]
    rerank_scores = reranker.predict(rerank_inputs)
    selected = [(c, float(score)) for c, score in zip(candidates, rerank_scores) if score >= req.min_score]
    selected = sorted(selected, key=lambda x: x[1], reverse=True)[:req.top_k]

    history = load_history()
    long_term_memory = {}
    last = history['topics'][-1] if history['topics'] else None

    if history['topics']:
        labels = [t['topic_label'] for t in history['topics']]
        label_embs = embedder.encode(labels)
        sim_scores = cosine_similarity(q_emb, label_embs)[0]
        best_idx = int(sim_scores.argmax())
        if sim_scores[best_idx] >= config.TOPIC_SIM_THRESHOLD:
            entry = history['topics'][best_idx]
            long_term_memory = entry

    retrieved_texts = [c['text'] for c, _ in selected]
    prompt_lines = [
        f"The user query is: {req.query}.",
        "Answer the user's question using the context below."
    ]

    if history['topics']:
        prompt_lines.append(f"Short-term memory for follow-up questions (Q is not the current turn's user query but the previously answered yser query): Q: {last['query']} A: {last['answer']}")
    if long_term_memory:
        prompt_lines.append(f"Long-term memory for previous discussed topics: {json.dumps(long_term_memory, indent=2)}")
    prompt_lines.append("\nChunks:\n" + "\n".join(retrieved_texts))
    

    full_prompt = "\n".join(prompt_lines)
    print(prompt_lines)
    raw_ans = tool.chat(model=config.LLAMA_MODEL, messages=[{"role": "user", "content": full_prompt}]).get("message", {}).get("content", "")
    answer = clean_text(raw_ans)

    topic_prompt = f"Q: {req.query}\nA: {answer}\n\nGive a short topic label."
    topic_resp = tool.chat(model=config.LLAMA_MODEL, messages=[{"role": "user", "content": topic_prompt}])
    topic_label = clean_text(topic_resp.get("message", {}).get("content", ""))

    history['topics'].append({"query": req.query, "answer": answer, "topic_label": topic_label})
    save_history(history)

    reranked_chunks = [RerankedChunk(
        text=clean_text(c['text']),
        score=score,
        source_url=c['source_url'],
        depth=c['depth']
    ) for c, score in selected]

    return QueryResponse(
        answer=answer,
        relevant_chunks=reranked_chunks,
        short_term_memory=f"Q: {last['query']} A: {last['answer']}" if last else "",
        long_term_memory=long_term_memory
    )