import json
import time

import requests
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

app = FastAPI()

OLLAMA_API_URL = "http://localhost:11434/api/generate"

def ollama_stream_to_openai_stream(prompt, model):
    data = {
        "model": model,
        "prompt": prompt,
        "stream": True
    }
    headers = {"Content-Type": "application/json"}
    try:
        response = requests.post(OLLAMA_API_URL, json=data, headers=headers, stream=True)
        response.raise_for_status()

        index = 0
        for line in response.iter_lines():
            if line:
                ollama_chunk = json.loads(line)
                content = ollama_chunk.get("response", "")
                # 构建符合OpenAI流式响应的JSON对象
                openai_chunk = {
                    "choices": [
                        {
                            "delta": {
                                "content": content
                            },
                            "finish_reason": None if not ollama_chunk.get("done") else "stop",
                            "index": index
                        }
                    ],
                    "created": time.time(),
                    "id": "chatcmpl-123",
                    "model": model,
                    "object": "chat.completion.chunk"
                }
                index += 1
                yield f"data: {json.dumps(openai_chunk)}\n\n"
        yield "data: [DONE]\n\n"
    except requests.RequestException as e:
        print(f"Request error: {e}")


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    data = await request.json()
    messages = data.get("messages", [])
    model = data.get("model", "mistral:7b")
    if not messages:
        return {"error": "No messages provided"}

    prompt = messages[-1].get("content", "")
    return StreamingResponse(ollama_stream_to_openai_stream(prompt, model), media_type="text/plain")


@app.get("/v1/models")
async def list_models():
    try:
        # 调用OpenAI的API获取模型列表
        response = ["mistral:7b"]
        return response
    except Exception as e:
        # 处理可能出现的异常
        return {"error": str(e)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
