# Day 40 — Streaming Responses
# Week 9 Day 1
#
# TOPIC 1: Why streaming matters
# client.messages.create() waits for the full response then sends it all at once.
# client.messages.stream() sends tokens as they're generated — text appears live.
# Key details: end="" stops print adding newlines between chunks; flush=True forces
# immediate display (Python buffers stdout by default); "with...as stream" is needed
# because stream() returns a context manager that opens/closes a server connection.

import os
import json
import datetime
import requests
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()
client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

# ============================================================
# TOPIC 2: Simple streaming call — no tools, just streaming text
# ============================================================

def stream_response(user_message):
    print("Agent: ", end="", flush=True)
    with client.messages.stream(
        model="claude-haiku-4-5",
        max_tokens=500,
        messages=[{"role": "user", "content": user_message}]
    ) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)
    print()


# ── Tool functions ──────────────────────────────────────────

def get_current_date():
    return datetime.date.today().isoformat()

def calculate(expression: str):
    allowed = set("0123456789+-*/().% ")
    if not all(c in allowed for c in expression):
        return "Error: invalid characters in expression"
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"

def get_weather(city: str):
    geo = requests.get(
        f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1"
    ).json()
    if not geo.get("results"):
        return f"City '{city}' not found."
    r = geo["results"][0]
    weather = requests.get(
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={r['latitude']}&longitude={r['longitude']}"
        f"&current=temperature_2m,windspeed_10m"
    ).json()
    current = weather["current"]
    return f"{city}, {r['country']}: {current['temperature_2m']}°C, wind {current['windspeed_10m']} km/h"

def get_country_info(country: str):
    headers = {"User-Agent": "PythonLearner/1.0"}
    response = requests.get(
        f"https://en.wikipedia.org/api/rest_v1/page/summary/{country.title()}",
        headers=headers
    )
    if response.status_code != 200:
        return f"Country '{country}' not found."
    return response.json()["extract"][:600]

# ── Tools list + dispatcher ─────────────────────────────────

tools = [
    {
        "name": "get_current_date",
        "description": "Returns today's date. Use for any question about current dates or deadlines.",
        "input_schema": {"type": "object", "properties": {}, "required": []}
    },
    {
        "name": "calculate",
        "description": "Evaluates a math expression. Use for ANY arithmetic — do NOT compute in your head.",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "A Python-style math expression"}
            },
            "required": ["expression"]
        }
    },
    {
        "name": "get_weather",
        "description": "Gets current weather for a city. Use for any weather or temperature question.",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name, e.g. 'Tokyo'"}
            },
            "required": ["city"]
        }
    },
    {
        "name": "get_country_info",
        "description": "Gets facts about a country — capital, population, language, currency. Use for any country question.",
        "input_schema": {
            "type": "object",
            "properties": {
                "country": {"type": "string", "description": "Country name, e.g. 'Japan'"}
            },
            "required": ["country"]
        }
    },
]

TOOL_MAP = {
    "get_current_date": lambda args: get_current_date(),
    "calculate":        lambda args: calculate(args["expression"]),
    "get_weather":      lambda args: get_weather(args["city"]),
    "get_country_info": lambda args: get_country_info(args["country"]),
}

# ============================================================
# TOPIC 3: Streaming inside the agent loop
# ============================================================
# Tool-use turns have no text — stream.text_stream yields nothing and the
# for loop does nothing. stream.get_final_message() gives the full response
# object so we can still check stop_reason and dispatch tool calls as normal.
# Only the final end_turn response has text, and that's what the user sees streaming.
def run_streaming_agent(user_message, max_turns=10):
    messages = [{"role": "user", "content": user_message}]

    for _ in range(max_turns):
        with client.messages.stream(
            model="claude-haiku-4-5",
            max_tokens=1024,
            tools=tools,
            messages=messages
        ) as stream:
            for text in stream.text_stream:
                print(text, end="", flush=True)
            final = stream.get_final_message()

        messages.append({"role": "assistant", "content": final.content})

        if final.stop_reason == "end_turn":
            print()
            return

        if final.stop_reason == "tool_use":
            tool_results = []
            for block in final.content:
                if block.type == "tool_use":
                    result = TOOL_MAP[block.name](block.input)
                    print(f"\n  [tool: {block.name} → {result}]")
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": str(result)
                    })
            messages.append({"role": "user", "content": tool_results})

    print("Error: agent did not complete within turn limit.")

if __name__ == "__main__":
    queries = [
        "Give me 3 quick facts about Tokyo.",
        "What's the weather in Lagos right now?",
    ]
    for query in queries:
        print(f"\nQ: {query}")
        print("A: ", end="", flush=True)
        run_streaming_agent(query)
        print("-" * 60)
