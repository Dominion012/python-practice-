# Day 43 — Prompt Engineering
# Week 9 Day 4
#
# TOPIC 1: A strong system prompt defines persona, task, constraints, and format
# in one place — not just one line. Claude follows it every turn, making the
# agent behave predictably. Weak prompts let Claude improvise; strong ones don't.
#
# TOPIC 2: Few-shot prompting shows Claude examples of input/output pairs instead
# of describing the task in words. Claude infers the pattern and applies it to
# new inputs — more reliable than instructions alone for strict format requirements.
#
# TOPIC 3: Chain of thought tells Claude to reason step by step before answering.
# On complex problems it catches errors in its own reasoning. It also makes
# outputs auditable — you can see exactly where Claude went right or wrong.

import os
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()
client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

# ============================================================
# TOPIC 1: System prompts done right
# ============================================================

MAYA_SYSTEM = """You are Maya, a customer service agent for TechFlow, a software company.

Your job:
- Answer questions about TechFlow products only
- Be friendly but concise — 2-3 sentences max
- Always end with "Is there anything else I can help you with?"

Never:
- Discuss competitors
- Make promises about refunds or pricing
- Answer questions unrelated to TechFlow
"""

def ask_maya(question):
    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=200,
        system=MAYA_SYSTEM,
        messages=[{"role": "user", "content": question}]
    )
    return response.content[0].text

# ============================================================
# TOPIC 2: Few-shot prompting — show don't tell
# ============================================================

FEW_SHOT_SYSTEM = """You classify customer messages into categories.

Examples:
Message: "My app keeps crashing when I open it"
Category: BUG

Message: "How do I reset my password?"
Category: SUPPORT

Message: "I want to cancel my subscription"
Category: BILLING

Message: "This product is amazing, thank you!"
Category: FEEDBACK

Reply with just the category word. Nothing else."""

def classify(message):
    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=20,
        system=FEW_SHOT_SYSTEM,
        messages=[{"role": "user", "content": message}]
    )
    return response.content[0].text.strip()

if __name__ == "__main__":
    questions = [
        "What does TechFlow do?",
        "Can you recommend a competitor?",
        "What's the weather like today?",
    ]
    for q in questions:
        print(f"Q: {q}")
        print(f"A: {ask_maya(q)}")
        print()

    print("--- Classifier ---")
    messages = [
        "The login button doesn't work",
        "How much does the pro plan cost?",
        "I love the new dashboard design",
        "Can you help me export my data?",
    ]
    for msg in messages:
        print(f"{classify(msg)} — {msg}")


# ============================================================
# TOPIC 3: Chain of thought — make Claude reason before answering
# ============================================================

def solve_without_cot(problem):
    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=100,
        messages=[{"role": "user", "content": problem}]
    )
    return response.content[0].text

def solve_with_cot(problem):
    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=400,
        system="Think through the problem step by step before giving your final answer. Show your reasoning.",
        messages=[{"role": "user", "content": problem}]
    )
    return response.content[0].text

if __name__ == "__main__":
    # ... keep existing code above ...

    print("--- Chain of Thought ---")
    problem = "A store sells apples for £1.20 each. If I buy 7 apples and pay with a £10 note, how much change do I get?"

    print("Without CoT:")
    print(solve_without_cot(problem))
    print()
    print("With CoT:")
    print(solve_with_cot(problem))
