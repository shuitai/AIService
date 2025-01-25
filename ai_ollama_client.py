import requests

OLLAMA_API_URL = "http://localhost:11434/api/generate"

data = {
    "model": "mistral:7b",
    "prompt": "who are you?",
    "stream": True
}
headers = {"Content-Type": "application/json"}

response = requests.post(OLLAMA_API_URL, json=data, headers=headers, stream=True)
response.raise_for_status()
for line in response.iter_lines():
    print(line)
