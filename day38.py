# Day 38 — Agents with Real External APIs
# Week 8 Day 3

import os
import datetime
import requests
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()
client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

# ============================================================
# TOPIC 1: The requests library — how to call any external API
# ============================================================
# requests.get(url).json() sends an HTTP GET and parses the JSON response.
# status 200 = success; anything else = error. Same library used in Day 35.

# ============================================================
# TOPIC 2: get_weather — two chained API calls
# ============================================================
# Weather APIs work with coordinates, not city names. So we geocode first
# (city name → lat/lon), then fetch weather at those coordinates.
# Both endpoints are from Open-Meteo: free, no API key required.
def get_weather(city: str):
    geo = requests.get(
        f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1"
    ).json()

    if not geo.get("results"):
        return f"City '{city}' not found."

    result = geo["results"][0]
    lat, lon, country = result["latitude"], result["longitude"], result["country"]

    weather = requests.get(
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        f"&current=temperature_2m,windspeed_10m"
    ).json()

    current = weather["current"]
    return (
        f"{city}, {country}: "
        f"{current['temperature_2m']}°C, "
        f"wind {current['windspeed_10m']} km/h"
    )

# ============================================================
# TOPIC 3: Full agent wired to a live API
# ============================================================
# Same tools/TOOL_MAP/run_agent pattern as Day 36 — the only difference is
# get_weather hits the internet instead of reading from a Python list.
# The agent autonomously decides to call get_weather twice for comparison
# questions, then reasons over both results to give a single answer.
tools = [
    {
        "name": "get_weather",
        "description": (
            "Gets the current weather for any city. Use whenever the user asks "
            "about weather, temperature, or climate in any location."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "The city name, e.g. 'Lagos' or 'London'"
                }
            },
            "required": ["city"]
        }
    }
]

TOOL_MAP = {
    "get_weather": lambda args: get_weather(args["city"]),
}

def run_agent(user_message, max_turns=10):
    messages = [{"role": "user", "content": user_message}]

    for _ in range(max_turns):
        response = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=500,
            tools=tools,
            messages=messages
        )
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            for block in response.content:
                if hasattr(block, "text"):
                    return block.text

        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    result = TOOL_MAP[block.name](block.input)
                    print(f"  -> {block.name}({block.input})")
                    print(f"     = {result}")
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": str(result)
                    })
            messages.append({"role": "user", "content": tool_results})

    return "Error: agent did not complete within the turn limit."

# three queries: single lookup, two-city comparison (agent calls tool twice),
# and a difference question (agent calls tool twice then computes the gap)
if __name__ == "__main__":
    queries = [
        "What's the weather in Lagos?",
        "Is it hotter in London or Paris right now?",
        "What's the temperature difference between New York and Lagos?",
    ]

    for query in queries:
        print(f"\nQ: {query}")
        answer = run_agent(query)
        print(f"A: {answer}")
        print("-" * 60)
