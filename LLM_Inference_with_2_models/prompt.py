import requests

url = "http://localhost:8000/generate"
data = {"prompt": "What is the capital of Portugal?"}

response = requests.post(url, json=data)
print(response.json())
