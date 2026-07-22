import os
import datetime
import requests
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()
client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

def get_country_info(country: str):
    headers = {"User-Agent": "PythonLearner/1.0"}
    response = requests.get(
        f"https://en.wikipedia.org/api/rest_v1/page/summary/{country.title()}",
        headers=headers
    )
    if response.status_code != 200:
        return f"Country '{country}' not found."
    data = response.json()
    return data["extract"][:600]



tools = [
    {
        "name": "get_country_info",
        "description": (
            "Gets facts about a country — capital, population, language, currency. "
            "Use when the user asks about any country."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "country": {
                    "type": "string",
                    "description": "The country name, e.g. 'Nigeria' or 'Japan'"
                }
            },
            "required": ["country"]
        }
    }
]

TOOL_MAP = {
    "get_country_info": lambda args: get_country_info(args["country"]),
}

def run_agent (user_input, max_turns = 5):
    messages = [{"role":"user", "content":user_input}]
    
    for _ in range(max_turns):
        response = client.messages.create(
            model = "claude-haiku-4-5",
            max_tokens= 500,
            tools= tools,
            messages = messages
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
                    tool_results.append({
                        "type" : "tool_result",
                        "tool_use_id" : block.id,
                        "content" : str(result)
                    })
            messages.append({"role":"user", "content": tool_results})


queries = [
    "What is the capital of Nigeria?",
    "What language do they speak in Brazil?",
    "What currency does Japan use?",
    "Which has a bigger population — Germany or France?",
]

if __name__ == "__main__":

    for query in queries:
        print(f"\nQ: {query}")
        reply = run_agent(query)
        print(reply)


