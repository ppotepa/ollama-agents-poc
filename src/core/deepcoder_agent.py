import os
import uvicorn
from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

from langchain_ollama import ChatOllama
from langchain.agents import initialize_agent, Tool

import uvicorn

# --- narzędzia ---
def write_file(filepath: str, contents: str) -> str:
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(contents)
    return f"✅ Zapisano do {filepath}"

def read_file(filepath: str) -> str:
    if not os.path.exists(filepath):
        return f"❌ Plik {filepath} nie istnieje"
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()

tools = [
    Tool(
        name="write_file",
        func=lambda x: write_file(**eval(x)),
        description="Zapisz plik. Argumenty: {'filepath': '...', 'contents': '...'}"
    ),
    Tool(
        name="read_file",
        func=lambda x: read_file(**eval(x)),
        description="Odczytaj plik. Argumenty: {'filepath': '...'}"
    ),
]

# --- model DeepCoder 14B ---
llm = ChatOllama(
    model="deepcoder:14b",
    temperature=0.2,
    num_ctx=8192,
)

# --- agent ---
agent = initialize_agent(
    tools,
    llm,
    agent="chat-zero-shot-react-description",
    verbose=True
)

# --- FastAPI serwer ---
app = FastAPI(title="DeepCoder Agent API")

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 512

@app.post("/v1/chat/completions")
async def chat_endpoint(request: ChatRequest):
    user_message = [m.content for m in request.messages if m.role == "user"][-1]
    try:
        result = agent.run(user_message)
        return {
            "id": "chatcmpl-deepcoder",
            "object": "chat.completion",
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": result},
                    "finish_reason": "stop",
                }
            ],
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/")
async def root():
    return {"status": "✅ DeepCoder Agent działa!"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
