import os
import time
import faiss
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
import numpy as np
import google.generativeai as genai

# Load environment variables (including your Google API key)
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize FastAPI
app = FastAPI(title="RAG System with Multiple User Guides")

# ---------- Helper Functions for Prompt Construction and Gemini Query ----------
def build_prompt(question: str, retrieved_chunks: list) -> str:
    """
    Build the prompt by including context from the retrieved chunks.
    """
    context = "\n\n".join(retrieved_chunks)
    prompt = (
        f"Use the following context from the user guides to answer the question.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )
    return prompt

def get_gemini_response(prompt: str) -> str:
    """Call the Gemini model using google.generativeai."""
    try:
        print("Invoking Gemini model...")
        model = genai.GenerativeModel("models/gemini-1.5-pro-latest")
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error from Gemini: {str(e)}")

# ---------- PDF Loading ----------
def load_pdf(filepath: str) -> str:
    """Load text from a PDF file."""
    reader = PdfReader(filepath)
    full_text = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            full_text.append(text)
    return "\n".join(full_text)

# ---------- Token-Based Chunking with Sliding Window ----------
def chunk_text_by_tokens_sliding(
    text: str,
    max_tokens: int = 500,
    stride: int = 100,
    tokenizer_name: str = "sentence-transformers/all-MiniLM-L6-v2"
):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokens = tokenizer.encode(text, add_special_tokens=False)

    chunks = []
    for start in range(0, len(tokens), stride):
        end = start + max_tokens
        chunk_tokens = tokens[start:end]
        if not chunk_tokens:
            break
        chunk_text = tokenizer.decode(chunk_tokens)
        chunks.append(chunk_text)
        if end >= len(tokens):
            break
    return chunks

# ---------- Load and Chunk PDFs ----------
# List of PDF files to include from the UserDocs folder
pdf_files = [
    "coresimUserGuide.pdf",
    "loadcoreUserGuide.pdf",
    "OranSimCeDeploymentGuide.pdf"
]

all_text = []
for pdf_file in pdf_files:
    pdf_path = os.path.join("UserDocs", pdf_file)
    if os.path.exists(pdf_path):
        print(f"Loading PDF from {pdf_path} ...")
        text = load_pdf(pdf_path)
        all_text.append(text)
    else:
        print(f"Warning: {pdf_path} does not exist.")

# Concatenate text from all available PDFs
combined_text = "\n\n".join(all_text)
chunks = chunk_text_by_tokens_sliding(combined_text, max_tokens=200, stride=100)
print(f"Total chunks created: {len(chunks)}")

# ---------- Embedding and Indexing ----------
embedder = SentenceTransformer('all-MiniLM-L6-v2')
print("Generating embeddings for chunks...")
chunk_embeddings = embedder.encode(chunks, convert_to_numpy=True)
faiss.normalize_L2(chunk_embeddings)

dimension = chunk_embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(chunk_embeddings)
print(f"FAISS index built with {index.ntotal} vectors.")

# ---------- API Endpoint ----------
class QueryRequest(BaseModel):
    question: str
    top_k: int = 3

@app.post("/ask")
async def ask_question(request: QueryRequest):
    # Generate an embedding for the question
    question_embedding = embedder.encode([request.question], convert_to_numpy=True)
    faiss.normalize_L2(question_embedding)
    
    # Retrieve the top_k similar chunks from the FAISS index
    distances, indices = index.search(question_embedding, request.top_k)
    retrieved_chunks = [chunks[i] for i in indices[0] if i < len(chunks)]
    
    # Build the full prompt with context and question
    full_prompt = build_prompt(request.question, retrieved_chunks)
    
    # Call Gemini for generating the answer
    start_time = time.time()
    answer = get_gemini_response(full_prompt)
    inference_time = time.time() - start_time

    return {
        "answer": answer,
        "retrieved_chunks": retrieved_chunks,
        "prompt": full_prompt,
        "inference_time": inference_time,
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
