# Day 36 — AI Agents with Tool Use
# Week 8 Day 1

# os reads env vars; datetime gives today's date; dotenv loads the .env file
import os
import datetime
from dotenv import load_dotenv
from anthropic import Anthropic

# load_dotenv() makes ANTHROPIC_API_KEY visible to os.environ; client is the object used for every API call
load_dotenv()
client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

# ============================================================
# TOPIC 1: Agents vs. simple LLM calls vs. RAG
# ============================================================
# Simple LLM call (Day 30/31): one round trip — model generates text from
# what it memorised during pretraining. Can't know today's date, can't
# look anything up, can hallucinate arithmetic.
#
# RAG (Day 32-33): developer always retrieves context and injects it —
# the model is still passive, it just gets extra text in its prompt.
#
# AGENT: the model DECIDES what tools to call and in what order, then loops
# until it has enough information to answer. The developer provides a
# toolkit; the model does the planning. Key shift: the model is driving.
# tools is the menu handed to Claude — each dict has name, description (when to use it), and input_schema (what args to supply)
tools = [
   {
    "name": "get_current_date",
    "description": (
        "Returns today's date as YYYY-MM-DD. Use whenever the user asks "
        "about current dates or deadlines relative to today — the model "
        "cannot know the current date without this tool."
    ),
    "input_schema": {
        "type": "object",
        "properties": {},   # no arguments needed
        "required": []
    }
  },


  {
      "name": "calculate",
      "description": (
        "Evaluates a math expression and returns the exact numeric result. "
        "Use for ANY arithmetic or percentages — do NOT compute in your head."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "A Python-style math expression, e.g. '15 / 100 * 8500'"
            }
        },
        "required": ["expression"]
    }
  },

  {
    "name" : "search_documents",
    "description": (
        "Searches the Acme Corp knowledge base for company policies. Use when the user asks about vacation, remote work, expenses, or office hours.."
    ),
    "input_schema" : {
        "type" : "object",
        "properties": {
            "query" :{
                "type": "string",
                "description": "The topic or question to search for, e.g. 'vacation days' or 'expense submission deadline. "
            }
        }, 
        "required" : ["query"]
    }
  }

]

# simple list of policy strings — search_documents filters this by keyword instead of embeddings
KNOWLEDGE_BASE = [
    "Vacations are only a max of 10 working days",
    "Remote work is only available to veterans",
    " Dining expenses are covered for the first 3 years of employment",
    "Vacation needs to be approved before allowed to leave",
    "Expense reports must be submitted within 30 days of the purchase date. Reports over $500 require manager approval.",

]

# returns today's date as a string like "2026-07-20" — model can't know this without calling the tool
def get_current_date():
    return datetime.date.today().isoformat()

# whitelist-only check before eval() prevents code injection; try/except catches division-by-zero etc.
def calculate(expression: str):
    allowed = set("0123456789+-*/().% ")
    if not all(c in allowed for c in expression):
        return "Error: invalid characters in expression"
    try:
        return str(eval(expression))
    except Exception as e :
        return f"Error {e}"

# splits query into words, keeps any KNOWLEDGE_BASE entry that contains at least one of them
def search_documents(query:str):
    words = query.lower().split()
    matches = [chunk for chunk in KNOWLEDGE_BASE
               if any(word in chunk.lower() for word in words )]
    return "\n".join (matches) if matches else "No relevant information found."



# dispatcher: maps tool name strings → actual functions so the loop can call any tool with one line
TOOL_MAP = {
    "get_current_date": lambda args: get_current_date(),
    "calculate":        lambda args: calculate(args["expression"]),
    "search_documents": lambda args: search_documents(args["query"]),
}


def run_agent(user_message, max_turns=10):
    # seed the conversation with the user's question; max_turns prevents infinite loops
    messages = [{"role": "user", "content": user_message}]

    for _ in range(max_turns):
        # pass tools= so Claude knows what it can call this turn
        response = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=500,
            tools=tools,
            messages=messages
        )
        # append the full content block (not just text) so Claude sees its own tool requests in history
        messages.append({"role": "assistant", "content": response.content })
        # end_turn means Claude is done — find the text block and return it
        if response.stop_reason == "end_turn":
            for block in response.content:
                if hasattr(block, "text"):
                    return block.text
        # tool_use means Claude wants to call one or more tools before answering
        if response.stop_reason == "tool_use":
           tool_results = []
           for block in response.content:
             if block.type == "tool_use":
               result = TOOL_MAP[block.name](block.input)
               print(f"  -> {block.name}({block.input})")
               print(f"     = {result}")
               # tool_use_id links this result back to the specific tool call Claude made
               tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": str(result)
            })
           # tool results go back as a "user" message — the API requires this role
           messages.append({"role": "user", "content": tool_results})


# four test queries covering: date lookup, arithmetic, single-tool retrieval, multi-tool chaining
if __name__ == "__main__":
    queries = [
        "What is today's date?",
        "If I earn $8,500 a month, what is 23% of my monthly salary?",
        "How many vacation days do I get?",
        "My expense report is dated today. What is the last day I can submit it?",
    ]

    for query in queries:
        print(f"\nQ: {query}")
        answer = run_agent(query)
        print(f"A: {answer}")
        print("-" * 60)
