import time
import os
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import google.generativeai as genai 
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()
# Configure Gemini API with API key from environment variable
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
print("🔑 GOOGLE_API_KEY:", os.getenv("GOOGLE_API_KEY"))

print("🔍 Available models:")
models = genai.list_models()
for model in models:
    print("-", model.name, model.supported_generation_methods)

def list_available_models():
    models = genai.list_models()
    for model in models:
        print(model.name, model.supported_generation_methods)




# FastAPI application instance
app = FastAPI()



# Ollama API setup
OLLAMA_URL = "http://localhost:11434/api/generate"  # Ollama default URL

class PromptRequest(BaseModel):
    prompt: str

@app.post("/generate")
async def generate_responses(request: PromptRequest):
    prompt = request.prompt

    # Measure time for Gemini response
    start_gemini = time.time()
    gemini_response = get_gemini_response(prompt)
    gemini_time = time.time() - start_gemini



    # Measure time for Ollama response
    start_ollama = time.time()
    ollama_response = get_ollama_response(prompt)
    ollama_time = time.time() - start_ollama

    # Return both responses along with inference times
    return {
    "gemini_response": gemini_response,
    "ollama_response": ollama_response,
    "gemini_inference_time": gemini_time,
    "ollama_inference_time": ollama_time,
    }


def get_gemini_response(prompt: str) -> str:
    try:
        print("✅ main.py loaded with model: models/gemini-1.5-pro-latest")
        print("Loading Gemini model:", "models/gemini-1.5-pro-latest")
        model = genai.GenerativeModel("models/gemini-1.5-pro-latest")
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error from Gemini: {str(e)}")


def get_ollama_response(prompt: str) -> str:
    headers = {"Content-Type": "application/json"}
    data = {
        "model": "llama2",  # make sure this matches what you pulled with ollama pull llama2
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "stream": False
    }

    try:
        response = requests.post("http://localhost:11434/api/chat", json=data, headers=headers)
        response.raise_for_status()
        return response.json().get("message", {}).get("content", "").strip()
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error from Ollama: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
