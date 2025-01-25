import requests

OPENAI_API_URL = "http://localhost:8000/v1/chat/completions"

data = {
    "messages": [
        {"role": "system", "content": "Always answer in rhymes."},
        {"role": "user", "content": "Introduce yourself."}
    ],
    "temperature": 0.7,
    "max_tokens": -1,
    "stream": True
}

headers = {"Content-Type": "application/json"}

response = requests.post(OPENAI_API_URL, json=data, headers=headers, stream=True)
response.raise_for_status()
for line in response.iter_lines():
    print(line)
