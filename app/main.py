from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.bot_logic import bot_instance  # Import the bot instance directly
from app.schemas import BotRequest, BotResponse

app = FastAPI(
    title="FinalBot Simple API",
    description="Simple API for your RAG-based bot with sources",
    version="1.0"
)

# Allow all origins for easy Android development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/ask", response_model=BotResponse)
async def ask_question(request: BotRequest):
    """Endpoint for getting answers from your RAG bot with sources"""
    try:
        # Get full response from bot instance
        bot_response = bot_instance.get_response(request.question)
        
        return {
            "question": request.question,
            "answer": bot_response["answer"],
            "sources": bot_response["sources"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ping")
async def health_check():
    """Simple health check endpoint"""
    return {"status": "ok", "message": "FinalBot is running"}