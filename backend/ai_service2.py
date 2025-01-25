import time

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
import httpx
import json

from starlette.requests import Request

app = FastAPI()

# Ollama 的 API 地址
OLLAMA_URL = "http://localhost:11434/api/generate"


# OpenAI 兼容的 Chat API 格式
async def generate_openai_compatible_response(prompt, model):
    """
    将用户输入的消息转换为 OpenAI 兼容的格式，并调用 Ollama 的 API。
    """
    # 构造 Ollama 的请求体
    ollama_data = {
        "model": model,
        "prompt": prompt,
        "stream": True
    }

    # 异步调用 Ollama 的 API
    async with httpx.AsyncClient() as client:
        async with client.stream("POST", OLLAMA_URL, json=ollama_data) as response:
            if response.status_code != 200:
                raise HTTPException(status_code=response.status_code, detail="Ollama API 调用失败")

            # 流式返回 OpenAI 兼容的格式
            async for chunk in response.aiter_lines():
                if chunk:
                    # 解析 Ollama 的响应
                    ollama_response = json.loads(chunk)
                    response_text = ollama_response.get("response", "")

                    # 构造 OpenAI 兼容的流式响应
                    openai_chunk = {
                        "id": "chatcmpl-123",  # 随机生成的 ID
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": response_text},
                                "finish_reason": None
                            }
                        ]
                    }

                    # 返回 JSON 格式的流式数据
                    yield f"data: {json.dumps(openai_chunk)}\n\n".encode('utf-8')

            # 返回结束标志
            yield "data: [DONE]\n\n".encode('utf-8')


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    data = await request.json()
    messages = data.get("messages", [])
    # model = data.get("model", "mistral:7b")
    model = "mistral:7b"
    if not messages:
        return {"error": "No messages provided"}

    prompt = messages[-1].get("content", "")
    return StreamingResponse(generate_openai_compatible_response(prompt, model), media_type="text/event-stream")


@app.get("/v1/models")
async def list_models():
    try:
        model = {
            "id": "openchat_3.6",
            "object": "model",
        }

        response = {
            "data": [model]
        }
        return response
    except Exception as e:
        return {"error": str(e)}


# 运行应用
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
