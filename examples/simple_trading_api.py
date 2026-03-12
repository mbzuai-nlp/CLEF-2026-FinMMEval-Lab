import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
from datetime import datetime
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI(title="Simple Trading API", version="1.0.0")

# Fixed trading prompt template
TRADING_PROMPT = """You are a professional swing trader specializing in daily trend forecasting.

Each day at UTC 00:01, you receive all information from the previous trading day, including market data, price movements, and relevant news. 
Your task is to analyze these signals and predict the likely price direction for the next day.

Current Date: {date}
Asset Symbol: {symbol}
Current Price: ${price}
Historical Prices (last 10 days): {historical_prices}

Recent News:
{news}

Instructions:
1. Focus on **short-term (1-day) direction** prediction.
2. Consider how yesterday’s data and sentiment affect **tomorrow’s expected move**.
3. Do not explain reasoning or include any text other than your final decision.

Possible outputs (choose ONE word only):
- BUY → if you expect the asset’s price to **increase tomorrow**
- HOLD → if you expect the asset’s price to **stay stable or are uncertain**
- SELL → if you expect the asset’s price to **decrease tomorrow**

Your decision (one word only):"""


class HistoricalPrice(BaseModel):
    date: str
    price: float


class TradingRequest(BaseModel):
    date: str
    price: Dict[str, float]  # {symbol: price}
    news: Dict[str, List[str]]  # {symbol: [news...]}
    symbol: List[str]  # [symbol]
    model: str
    history_price: Dict[str, List[Dict]] = {}  # {symbol: [{"date": "...", "price": ...}]}


class TradingResponse(BaseModel):
    message: str
    timestamp: str
    price: Dict[str, float]
    news_count: int
    recommended_action: str


def call_openai(prompt: str, model: str) -> str:
    """Call OpenAI API"""
    import openai
    
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=10
    )
    
    return response.choices[0].message.content.strip().upper()


def call_anthropic(prompt: str, model: str) -> str:
    """Call Anthropic Claude API"""
    import anthropic
    
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    
    response = client.messages.create(
        model=model,
        max_tokens=10,
        temperature=0.7,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    
    return response.content[0].text.strip().upper()


def call_gemini(prompt: str, model: str) -> str:
    """Call Google Gemini API"""
    import google.generativeai as genai
    
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    
    model_instance = genai.GenerativeModel(model)
    response = model_instance.generate_content(
        prompt,
        generation_config={
            "temperature": 0.7,
            "max_output_tokens": 10,
        }
    )
    
    return response.text.strip().upper()


def call_deepseek(prompt: str, model: str, endpoint: str = None) -> str:
    """Call DeepSeek API (OpenAI-compatible)"""
    import openai
    
    # Use custom endpoint if provided, otherwise use default
    if endpoint is None:
        endpoint = os.getenv("DEEPSEEK_ENDPOINT", "https://api.deepseek.com/v1")
    
    client = openai.OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=endpoint
    )
    
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=10
    )
    
    return response.choices[0].message.content.strip().upper()


def call_together(prompt: str, model: str) -> str:
    """Call Together AI API"""
    import openai
    
    client = openai.OpenAI(
        api_key=os.getenv("TOGETHER_API_KEY"),
        base_url="https://api.together.xyz/v1"
    )
    
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt}
        ],
    )
    
    return response.choices[0].message.content.strip().upper()


def get_llm_decision(prompt: str) -> str:
    """Route to appropriate LLM provider based on model name"""

    model = "gpt-4o"
    # OpenAI models
    if model.startswith("gpt-") or model.startswith("o1-") or model.startswith("o3"):
        return call_openai(prompt, model)
    
    # Anthropic models
    elif model.startswith("claude-"):
        return call_anthropic(prompt, model)
    
    # Gemini models
    elif model.startswith("gemini-"):
        return call_gemini(prompt, model)
    
    # Together AI models (includes DeepSeek, Qwen, etc.)
    elif model.startswith("deepseek-") or model.startswith("qwen"):
        return call_together(prompt, model)

    else:
        raise ValueError(f"Unsupported model: {model}")


@app.get("/")
async def home():
    return {"message": "Simple Trading API - LLM-based trading decisions"}


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/trading_action/")
async def get_trading_decision(request: TradingRequest):
    """
    Get trading decision from LLM
    
    Example request:
    {
        "date": "2025-10-25",
        "price": {"BTC": 67500.50},
        "news": {
            "BTC": [
                "Bitcoin reaches new all-time high",
                "Major institution announces Bitcoin adoption"
            ]
        },
        "symbol": ["BTC"],
        "history_price": {
            "BTC": [
                {"date": "2025-10-15", "price": 65000},
                {"date": "2025-10-16", "price": 65500}
            ]
        }
    }
    """
    try:
        # Extract symbol (we only handle single symbol)
        if not request.symbol or len(request.symbol) == 0:
            raise HTTPException(status_code=400, detail="No symbol provided")
        
        symbol = request.symbol[0]
        
        # Get price for this symbol
        if symbol not in request.price:
            raise HTTPException(status_code=400, detail=f"No price for symbol {symbol}")
        price = request.price[symbol]
        
        # Get news for this symbol
        news_list = request.news.get(symbol, [])
        news_text = "\n".join([f"- {n}" for n in news_list]) if news_list else "No recent news"
        
        # Get historical prices for this symbol
        history_data = request.history_price.get(symbol, [])
        if history_data:
            hist_prices_text = ", ".join([f"${item['price']:.2f}" for item in history_data])
        else:
            hist_prices_text = "No historical data available"
        
        # Build prompt
        prompt = TRADING_PROMPT.format(
            date=request.date,
            symbol=symbol,
            price=price,
            historical_prices=hist_prices_text,
            news=news_text
        )
        
        print(f"\n{'='*60}")
        print(f"Symbol: {symbol}")
        print(f"Date: {request.date}")
        print(f"Price: ${price}")
        print(f"History Price Points: {len(history_data)}")
        print(f"News Count: {len(news_list)}")
        print(f"{'='*60}\n")
        
        # Get LLM decision
        raw_decision = get_llm_decision(prompt)
        
        # Extract action (handle cases where LLM returns extra text)
        action = "HOLD"  # Default
        if "BUY" in raw_decision:
            action = "BUY"
        elif "SELL" in raw_decision:
            action = "SELL"
        elif "HOLD" in raw_decision:
            action = "HOLD"
        
        print(f"Raw LLM Response: {raw_decision}")
        print(f"Extracted Action: {action}")
        print(f"{'='*60}\n")
        
        return TradingResponse(
            message="Data received successfully",
            timestamp=datetime.now().isoformat(),
            price=request.price,
            news_count=len(news_list),
            recommended_action=action
        )
        
    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get trading decision: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    print("Starting Simple Trading API...")
    print("Supported models:")
    print("  - OpenAI: gpt-4o, gpt-4.1, gpt-4.1-mini, o1-preview, o3")
    print("  - Anthropic: claude-3-5-sonnet-20241022, claude-3-5-haiku-20241022, claude-sonnet-4-20250514")
    print("  - Google: gemini-2.0-flash, gemini-2.5-pro-preview-06-05")
    print("  - Together AI: deepseek-v3-1, deepseek-v3, deepseek-r1, qwen3-235b")
    print("\nAPI will be available at: http://0.0.0.0:62237")
    
    uvicorn.run(app, host="0.0.0.0", port=62237, log_level="info")

