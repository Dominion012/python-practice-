# Day 31 — Prompt Engineering
# Week 6 Day 6

import os
import json
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()
client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

def ask(prompt, max_tokens=150, temperature=1):
    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text

# ============================================================
# TOPIC 1: Why Prompt Engineering Matters
# ============================================================
# Same model, same weights — the prompt alone steers what's "statistically
# likely" to come next. Specific, constrained prompts produce more usable
# AND cheaper output: a vague prompt rambles with no natural stopping point
# (can hit max_tokens before finishing), a precise one answers exactly what
# was asked and stops cleanly.
#
# vague   = ask("Tell me about Python lists.")
# precise = ask("In exactly 2 sentences, explain the one key difference "
#                "between a Python list and a tuple, using a one-line code example.")

# ============================================================
# TOPIC 2: System Prompts vs. User Prompts
# ============================================================
# system= sets persistent, developer-controlled behavior across the whole
# conversation. messages= is the actual back-and-forth with the end user.
# Models are trained (via RLHF) to trust system instructions over user text
# specifically to resist "prompt injection" — a user trying to override the
# app's rules via crafted input.

def ask_with_system(system_prompt, user_prompt, max_tokens=150):
    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=max_tokens,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}]
    )
    return response.content[0].text

# pirate = ask_with_system("You always respond like a pirate, regardless of topic.", "What is a Python dictionary?")
# formal = ask_with_system("You are a formal computer science professor.", "What is a Python dictionary?")

# ============================================================
# TOPIC 3: Zero-Shot vs. Few-Shot Prompting
# ============================================================
# Few-shot examples teach the model an exact output format via in-context
# learning — no weights change, it's pattern-matching within one forward
# pass. Critical for programmatic use: a few-shot prompt returns one clean
# parseable word; zero-shot tends to hedge with a full explanation.

FEW_SHOT_SENTIMENT_TEMPLATE = """Classify the sentiment as exactly one word: POSITIVE, NEGATIVE, or MIXED.

Review: "Best purchase I've made all year!"
Sentiment: POSITIVE

Review: "Total waste of money, broke in a week."
Sentiment: NEGATIVE

Review: "Great design but terrible battery life."
Sentiment: MIXED

Review: "{review_text}"
Sentiment:"""

# zero_shot = ask('Classify the sentiment of this review: "The movie was a rollercoaster, in the worst way."')
# few_shot  = ask(FEW_SHOT_SENTIMENT_TEMPLATE.format(review_text="The movie was a rollercoaster, in the worst way."), max_tokens=10)

# ============================================================
# TOPIC 4: Chain-of-Thought Prompting
# ============================================================
# Generation is autoregressive (Day 30) — every written token becomes part
# of the context for the next one. Asking the model to show its reasoning
# gives it a "scratchpad" to lean on for the final answer, rather than
# requiring the correct answer in its very first tokens with no working-out.

BAT_AND_BALL = "A bat and ball together cost $1.10. The bat costs $1.00 more than the ball. How much does the ball cost?"

# direct = ask(f"{BAT_AND_BALL} Answer with just the dollar amount, nothing else.", max_tokens=10)
# cot    = ask(f"{BAT_AND_BALL} Think through this step by step, showing your reasoning, then give the final answer.", max_tokens=200)

# ============================================================
# TOPIC 5: Reusable Prompt Templates
# ============================================================
# Separate the fixed prompt structure from per-call data via a template
# function — same input data varies, zero code changes, same pattern as
# Day 29's Flask route handling many different requests.

def build_sentiment_prompt(review_text):
    return FEW_SHOT_SENTIMENT_TEMPLATE.format(review_text=review_text)

# reviews = [
#     "This changed my life, absolutely incredible.",
#     "Arrived broken and customer service ignored me.",
#     "Works well but the price is way too high for what you get.",
#     "This was a night to remember",
# ]
# for review in reviews:
#     print(f"{review!r} -> {ask(build_sentiment_prompt(review), max_tokens=10).strip()}")

# ============================================================
# TOPIC 6: Structured Outputs — Reliable JSON
# ============================================================
# LLMs sometimes wrap JSON in markdown fences despite being told not to —
# strip those defensively before json.loads(), or it crashes with
# "Expecting value: line 1 column 0" on real responses.

def strip_markdown_fence(text):
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        text = text.rsplit("```", 1)[0]
    return text.strip()

def extract_info(text):
    prompt = f"""Extract the following information from this customer message and respond with ONLY valid JSON, no other text, no markdown formatting:

{{
  "customer_name": string or null,
  "issue_category": one of "billing", "technical", "shipping", "other",
  "urgency": one of "low", "medium", "high"
}}

Customer message: "{text}"
"""
    raw = ask(prompt, max_tokens=150, temperature=0)
    return json.loads(strip_markdown_fence(raw))

# ============================================================
# TOPIC 7: Temperature — Controlling Randomness
# ============================================================
# softmax(logits / temperature). temperature=0 is near-deterministic (same
# input -> same output, every time) — the right choice for extraction and
# classification. Higher temperature flattens the distribution for more
# variety — useful for creative tasks, wrong choice for anything your code
# needs to parse reliably.

# print("TEMPERATURE = 0 (3 runs):")
# for _ in range(3):
#     print(f"  {ask('Write a one-sentence tagline for a coffee shop.', max_tokens=50, temperature=0)}")
#
# print("TEMPERATURE = 1 (3 runs):")
# for _ in range(3):
#     print(f"  {ask('Write a one-sentence tagline for a coffee shop.', max_tokens=50, temperature=1)}")

# ============================================================
# TOPIC 8: Capstone — Customer Support Triage Tool
# ============================================================

def triage_ticket(message):
    prompt = f"""You are a customer support triage assistant. Extract structured information from this message and respond with ONLY valid JSON, no markdown formatting:

{{
  "customer_name": string or null,
  "issue_category": one of "billing", "technical", "shipping", "other",
  "urgency": one of "low", "medium", "high",
  "suggested_response": a one-sentence draft reply acknowledging the issue
}}

Customer message: "{message}"
"""
    raw = ask(prompt, max_tokens=300, temperature=0)
    return json.loads(strip_markdown_fence(raw))

if __name__ == "__main__":
    print("=== Customer Support Triage Tool ===")
    while True:
        user_input = input("\nPaste a customer message (or 'quit'): ")
        if user_input.lower() == "quit":
            break
        ticket = triage_ticket(user_input)
        print(f"\n  Customer: {ticket['customer_name']}")
        print(f"  Category: {ticket['issue_category']}")
        print(f"  Urgency:  {ticket['urgency']}")
        print(f"  Suggested reply: {ticket['suggested_response']}")
