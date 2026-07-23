# Day 39 — Week 8 Capstone: Travel Planning Agent
# Week 8 Day 4
#
# Combines everything from the week:
# - Day 36: agentic loop + TOOL_MAP dispatch
# - Day 37: ConversationalAgent class with persistent self.messages
# - Day 38: real external API calls (weather + Wikipedia)
# All 4 tools (date, calculate, weather, country info) live in one agent.
# system= sets a permanent instruction Claude follows every turn — used here
# to enforce short answers and remind it to remember trip details.

import os
import json
import datetime
import requests
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()
client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

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

# ── TravelAgent class ──────────────────────────────────────
# Same pattern as Day 37's ConversationalAgent: self.messages grows across
# turns so the agent remembers "Japan" from message 1 on message 4.
# save/load persists the conversation to disk across restarts.
class TravelAgent:
    def __init__(self):
        self.messages = []

    def chat(self, user_message, max_turns=10):
        self.messages.append({"role": "user", "content": user_message})

        for _ in range(max_turns):
            response = client.messages.create(
                model="claude-haiku-4-5",
                max_tokens=1024,
                tools=tools,
                messages=self.messages,
                system="You are a travel planning assistant. Give short, direct answers — one or two sentences max. Remember trip details the user mentions."
            )
            self.messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason == "end_turn":
                for block in response.content:
                    if hasattr(block, "text"):
                        return block.text

            if response.stop_reason == "tool_use":
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        result = TOOL_MAP[block.name](block.input)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": str(result)
                        })
                self.messages.append({"role": "user", "content": tool_results})

        return "Error: agent did not complete within the turn limit."

    def save(self, filepath="travel_agent.json"):
        serializable = []
        for msg in self.messages:
            if isinstance(msg["content"], list):
                content = [b.model_dump() if hasattr(b, "model_dump") else b for b in msg["content"]]
                serializable.append({"role": msg["role"], "content": content})
            else:
                serializable.append(msg)
        with open(filepath, "w") as f:
            json.dump(serializable, f, indent=2)

    def load(self, filepath="travel_agent.json"):
        if os.path.exists(filepath):
            try:
                with open(filepath) as f:
                    self.messages = json.load(f)
            except json.JSONDecodeError:
                print("Warning: save file corrupted. Starting fresh.")
                self.messages = []

if __name__ == "__main__":
    agent = TravelAgent()
    agent.load()

    print("Travel Planning Assistant — type 'quit' to exit, 'save' to save.\n")
    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() == "quit":
            break
        if user_input.lower() == "save":
            agent.save()
            print("Conversation saved.\n")
            continue
        answer = agent.chat(user_input)
        print(f"Agent: {answer}\n")
