Have a router route api requests to local ollama (llama2-7b) and cloud based gemini and compare inference time.

(genai_env) PS C:\GenAI\projects\project1> uvicorn main:app --host 0.0.0.0 --port 8000 --reload

(genai_env) PS C:\GenAI\projects\project1>ollama serve
(genai_env) PS C:\GenAI\projects\project1>ollama run llama2

(genai_env) PS C:\GenAI\projects\project1> python .\prompt.py
{'gemini_response': 'Lisbon', 'ollama_response': 'The capital city of Portugal is Lisbon (Portuguese: Lisboa).', 'gemini_inference_time': 0.6318953037261963, 'ollama_inference_time': 4.297216415405273}