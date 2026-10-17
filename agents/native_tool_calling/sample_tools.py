"""
Demostration tools with strict Pydantic schemas:
- get_stock_price: Intentionally raises ValueError on unknown tickers (tests error handling).
- calculate: Safe mathematical expression evaluator.
- get_weather: Weather report simulator.
"""


from typing import Literal
from pydantic import BaseModel, Field
from agents.native_tool_calling.tool_registry import ToolRegistry

# Instantiate global registry
registry = ToolRegistry()

#* Tool 1: Stock Price (Supports error testing)
class StockPriceArgs(BaseModel):
    ticker: str = Field(description="The stock ticker symbol in uppercase, e.g. AAPL, NVDA, TSLA")

@registry.register(
    name="get_stock_price",
    description="Get the current real-time stock price for a compant ticker.",
    args_schema=StockPriceArgs
)
def get_stock_price(ticker: str) -> dict:
    prices = {
        "APPL": 224.50,
        "NVDA": 138.33,
        "GOOGL": 182.54,
        "MSFT": 448.02,
        "AMZN": 3724.33,
        "TSLA": 255.53
    }
    ticker_clean = ticker.upper().strip()
    if ticker_clean not in prices:
        # raise ValueError(f"Ticker '{ticker_clean}' not found in database. Supported tickers: {', '.join(prices.keys())}")
        
        # Removed the supported list to check the The "Trivial Tweak" Trap (Near-Identical Cycling)
        raise ValueError(f"Ticker '{ticker_clean}' not found in database.")
    return {
        "ticker": ticker_clean,
        "price_usd": prices[ticker_clean]
    }


#* Tool 2: Calculator (Safe mathematical expression evaluator)
class CalculatorArgs(BaseModel):
    expression: str = Field(description="Mathematical expression to evaluate, e.g. '224.50 + 138.25' or '150 * 1.18'")

@registry.register(
    name="calculate",
    description="Evaluate arithmetic expressions. Supports +, -, *, /, and parentheses.",
    args_schema=CalculatorArgs
)
def calculate(expression: str) -> float:
    # Whitelist allowed characters for safe evaluation
    allowed_chars = set("0123456789+-*/(). ")
    if not set(expression).issubset(allowed_chars):
        raise ValueError("Invalid mathematical expression. Only basic arithmetic operators are allowed.")
    return float(eval(expression, {"__builtins__": None}, {}))


#* Tool 3: Weather (Simulated)
class WeatherArgs(BaseModel):
    city: str = Field(description="The city name, e.g. 'Hyderabad', 'London', 'Tokyo'")
    unit: Literal["celsius", "fahrenheit"] = Field(
        default="celsius",
        description="Temperature unit to return"
    )

@registry.register(
    name="get_weather",
    description="Retrieve the current weather and temperature for a given city.",
    args_schema=WeatherArgs
)
def get_weather(city: str, unit: str = "celsius") -> dict:
    weather_data = {
        "hyderabad": {"temp_c": 31, "condition": "Sunny"},
        "london": {"temp_c": 14, "condition": "Light Rain"},
        "tokyo": {"temp_c": 19, "condition": "Clear"},
    }
    city_lower = city.lower().strip()
    data = weather_data.get(city_lower, {"temp_c": 22, "condition": "Partly Cloudy"})
    
    temp = data["temp_c"]
    if unit == "fahrenheit":
        temp = round((temp * 9/5) + 32, 1)
    return {"city": city.title(), "temperature": temp, "unit": unit, "condition": data["condition"]}
