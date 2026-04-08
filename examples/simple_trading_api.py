import os
from typing import Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

load_dotenv()

app = FastAPI(title="Simple Trading API", version="2.0.0")

DEFAULT_MODEL = os.getenv("TRADING_MODEL", "gpt-4o")

TRADING_PROMPT = """You are a professional swing trader specializing in daily trend forecasting.

Each day at UTC 00:01, you receive all information from the previous trading day, including market data, price movements, and relevant news.
Your task is to analyze these signals and predict the likely price direction for the next day.

Current Date: {date}
Asset Symbol: {symbol}
Current Price: ${price}
Historical Prices (last 10 days): {historical_prices}
Momentum Signal: {momentum}

Recent News:
{news}

Recent 10-K Context:
{ten_k}

Recent 10-Q Context:
{ten_q}

Instructions:
1. Focus on short-term (1-day) direction prediction.
2. Consider price history, momentum, yesterday's news, and any filing context when available.
3. Output exactly one word and nothing else.

Possible outputs:
- BUY
- HOLD
- SELL

Your decision:"""


class HistoricalPrice(BaseModel):
    date: str
    price: float


class TradingRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    date: str
    price: Dict[str, float]
    news: Dict[str, List[str]]
    symbol: List[str]
    momentum: Optional[Dict[str, str]] = None
    history_price: Dict[str, List[HistoricalPrice]] = Field(default_factory=dict)
    ten_k: Optional[Dict[str, List[str]]] = Field(default=None, alias="10k")
    ten_q: Optional[Dict[str, List[str]]] = Field(default=None, alias="10q")


class TradingResponse(BaseModel):
    recommended_action: str


def call_openai(prompt: str, model: str) -> str:
    import openai

    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=10,
    )
    return response.choices[0].message.content.strip().upper()


def call_anthropic(prompt: str, model: str) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    response = client.messages.create(
        model=model,
        max_tokens=10,
        temperature=0.2,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text.strip().upper()


def call_gemini(prompt: str, model: str) -> str:
    import google.generativeai as genai

    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    model_instance = genai.GenerativeModel(model)
    response = model_instance.generate_content(
        prompt,
        generation_config={"temperature": 0.2, "max_output_tokens": 10},
    )
    return response.text.strip().upper()


def call_together(prompt: str, model: str) -> str:
    import openai

    client = openai.OpenAI(
        api_key=os.getenv("TOGETHER_API_KEY"),
        base_url="https://api.together.xyz/v1",
    )
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=10,
    )
    return response.choices[0].message.content.strip().upper()


def get_llm_decision(prompt: str, model: str = DEFAULT_MODEL) -> str:
    if model.startswith(("gpt-", "o1-", "o3")):
        return call_openai(prompt, model)
    if model.startswith("claude-"):
        return call_anthropic(prompt, model)
    if model.startswith("gemini-"):
        return call_gemini(prompt, model)
    if model.startswith(("deepseek-", "qwen")):
        return call_together(prompt, model)
    raise ValueError(f"Unsupported model: {model}")


def join_context(items: Optional[List[str]], fallback: str) -> str:
    if not items:
        return fallback
    return "\n".join(f"- {item}" for item in items)


def extract_action(raw_decision: str) -> str:
    if "BUY" in raw_decision:
        return "BUY"
    if "SELL" in raw_decision:
        return "SELL"
    return "HOLD"


@app.get("/")
async def home():
    return {"message": "Simple Trading API example for FinMMEval Task 3"}


@app.get("/health")
async def health():
    return {"status": "healthy", "model": DEFAULT_MODEL}


@app.post("/trading_action/", response_model=TradingResponse)
async def get_trading_decision(request: TradingRequest):
    try:
        if not request.symbol:
            raise HTTPException(status_code=400, detail="No symbol provided")

        symbol = request.symbol[0]
        if symbol not in request.price:
            raise HTTPException(status_code=400, detail=f"No price for symbol {symbol}")

        price = request.price[symbol]
        news_text = join_context(request.news.get(symbol), "No recent news")
        momentum = (request.momentum or {}).get(symbol, "No momentum signal provided")
        history = request.history_price.get(symbol, [])
        history_text = ", ".join(f"${item.price:.2f}" for item in history) if history else "No historical data available"
        ten_k_text = join_context((request.ten_k or {}).get(symbol), "No 10-K context provided")
        ten_q_text = join_context((request.ten_q or {}).get(symbol), "No 10-Q context provided")

        prompt = TRADING_PROMPT.format(
            date=request.date,
            symbol=symbol,
            price=price,
            historical_prices=history_text,
            momentum=momentum,
            news=news_text,
            ten_k=ten_k_text,
            ten_q=ten_q_text,
        )

        raw_decision = get_llm_decision(prompt)
        return TradingResponse(recommended_action=extract_action(raw_decision))
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to get trading decision: {exc}") from exc


if __name__ == "__main__":
    import uvicorn

    print("Starting Simple Trading API example...")
    print(f"Using model: {DEFAULT_MODEL}")
    print("API will be available at: http://0.0.0.0:62237")
    uvicorn.run(app, host="0.0.0.0", port=62237, log_level="info")
