# Day 37 — Conversational Agents with Memory
# Week 8 Day 2

import os
import json
import datetime
from dotenv import load_dotenv
from anthropic import Anthropic

# load_dotenv() reads .env; client is the object used for every API call
load_dotenv()
client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

# ============================================================
# TOPIC 1: The memory problem — Day 36's agent forgot everything between calls
# ============================================================
# run_agent() created messages = [] fresh each call and threw it away on return.
# Fix: move messages into a class instance so it persists across turns.
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




# ============================================================
# TOPIC 2: ConversationalAgent class
# ============================================================
# Each instance owns its own self.messages list — two agents never share history.
# chat() adds to self.messages instead of resetting it, so the agent sees the
# full conversation history on every API call.
class ConversationalAgent:
    def __init__(self):
        self.messages = []
        self.tools = tools

    def chat(self, user_message, max_turns=10):
        self.messages.append({"role": "user", "content": user_message})

        for _ in range(max_turns):
            response = client.messages.create(
                model="claude-haiku-4-5",
                max_tokens=1024,
                tools=self.tools,
                messages=self.messages
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

    # ============================================================
    # TOPIC 3: Saving and loading conversation history
    # ============================================================
    def save(self, filepath="conversation.json"):
        # API response blocks are SDK objects (TextBlock, ToolUseBlock) — json.dump can't write them
        # model_dump() converts them to plain dicts that JSON can handle
        serializable = []
        for msg in self.messages:
            if isinstance(msg["content"], list):
                content = []
                for block in msg["content"]:
                    if hasattr(block, "model_dump"):
                        content.append(block.model_dump())
                    else:
                        content.append(block)
                serializable.append({"role": msg["role"], "content": content})
            else:
                serializable.append(msg)
        with open(filepath, "w") as f:
            json.dump(serializable, f, indent=2)

    def load(self, filepath="conversation.json"):
        # os.path.exists guard prevents a crash on the very first run when no save file exists yet
        if os.path.exists(filepath):
            with open(filepath) as f:
                self.messages = json.load(f)


# while True REPL: keeps asking for input until "quit"; "save" persists history to disk
if __name__ == "__main__":
    agent = ConversationalAgent()
    agent.load()  # picks up previous conversation if one was saved

    print("Acme Corp Agent — type 'quit' to exit, 'save' to save conversation.\n")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "quit":
            break
        if user_input.lower() == "save":
            agent.save()
            print("Conversation saved.\n")
            continue
        answer = agent.chat(user_input)
        print(f"Agent: {answer}\n")
