from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
from contextlib import asynccontextmanager
from fatcat_gpt import FatCatGPT

bot = None

# =========================================================
# start server
# =========================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global bot
    print("[伺服器啟動中] 正在將模型載入到背景，請稍候...")
    bot = FatCatGPT()
    print("[伺服器已就緒] 胖貓模型載入完成，等待呼叫！")
    yield
    print("[伺服器關閉中] 正在釋放顯示卡記憶體...")
    bot = None

app = FastAPI(title="FatCat GPT API", lifespan=lifespan)

class ChatRequest(BaseModel):
    messages: List[Dict[str, str]]
    max_new_tokens: int = 128
    temperature: float = 0.3
    top_p: float = 0.95

# =========================================================
# Endpoint: /chat
# =========================================================
@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    if bot is None:
        raise HTTPException(status_code=500, detail="模型尚未載入完成")
    
    try:
        response = bot.chat(
            messages=request.messages,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p
        )
        return {"status": "success", "response": response}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))