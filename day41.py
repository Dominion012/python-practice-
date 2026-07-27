# Day 41 — Structured Outputs
# Week 9 Day 2
#
# TOPIC 1: Asking Claude for JSON without enforcement often fails — it wraps
# the output in markdown fences (```json ... ```) or adds explanation text,
# so json.loads() raises JSONDecodeError. You cannot reliably parse it.
#
# TOPIC 2: Tool forcing fixes this. Pass tool_choice={"type": "tool", "name": "..."}
# and Claude MUST call that tool — no markdown, no prose. block.input is already
# a clean Python dict; no json.loads() needed at all.
#
# TOPIC 3: The same pattern works for any schema. Define the shape of data you
# want (sentiment score, job salary, boolean remote field) and Claude fills it.
# The tool never executes — the input_schema IS the output format.

import os
import json
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()
client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

# ============================================================
# TOPIC 1: The unstructured output problem
# ============================================================

text = "My name is Domi, I'm 23 years old, and my email is domi@example.com"

response = client.messages.create(
    model="claude-haiku-4-5",
    max_tokens=200,
    messages=[{
        "role": "user",
        "content": f"Extract the name, age, and email from this text as JSON: {text}"
    }]
)

raw = response.content[0].text
print("Raw output:")
print(raw)
print()

try:
    data = json.loads(raw)
    print("Parsed successfully:", data)
except json.JSONDecodeError:
    print("FAILED to parse — Claude wrapped it in markdown or added extra text")

# ============================================================
# TOPIC 2: Tool forcing — guaranteed structured output
# ============================================================

extract_tool = {
    "name": "extract_contact",
    "description": "Extract contact information from text.",
    "input_schema": {
        "type": "object",
        "properties": {
            "name":  {"type": "string", "description": "Full name"},
            "age":   {"type": "integer", "description": "Age in years"},
            "email": {"type": "string", "description": "Email address"}
        },
        "required": ["name", "age", "email"]
    }
}

response = client.messages.create(
    model="claude-haiku-4-5",
    max_tokens=200,
    tools=[extract_tool],
    tool_choice={"type": "tool", "name": "extract_contact"},
    messages=[{
        "role": "user",
        "content": f"Extract the contact info from this text: {text}"
    }]
)

for block in response.content:
    if block.type == "tool_use":
        data = block.input
        print("Structured output:", data)
        print(f"Name:  {data['name']}")
        print(f"Age:   {data['age']}")
        print(f"Email: {data['email']}")

# ============================================================
# TOPIC 3: Practical extractions — same pattern, different schemas
# ============================================================

# ── Example 1: Sentiment analysis ──────────────────────────
review = "The food was amazing but the service was incredibly slow. Waited 40 minutes for our order."

sentiment_tool = {
    "name": "analyze_sentiment",
    "description": "Analyze the sentiment of a customer review.",
    "input_schema": {
        "type": "object",
        "properties": {
            "sentiment": {"type": "string", "description": "positive, negative, or mixed"},
            "score":     {"type": "integer", "description": "1-10 satisfaction score"},
            "summary":   {"type": "string", "description": "One sentence summary of the review"}
        },
        "required": ["sentiment", "score", "summary"]
    }
}

response = client.messages.create(
    model="claude-haiku-4-5",
    max_tokens=200,
    tools=[sentiment_tool],
    tool_choice={"type": "tool", "name": "analyze_sentiment"},
    messages=[{"role": "user", "content": f"Analyze this review: {review}"}]
)

for block in response.content:
    if block.type == "tool_use":
        print("Sentiment analysis:", block.input)

# ── Example 2: Job posting extractor ───────────────────────
job_post = "Senior Python Engineer at Stripe. Remote-friendly. Salary: $150k-$200k. Must have 5+ years experience."

job_tool = {
    "name": "extract_job",
    "description": "Extract structured data from a job posting.",
    "input_schema": {
        "type": "object",
        "properties": {
            "title":      {"type": "string",  "description": "Job title"},
            "company":    {"type": "string",  "description": "Company name"},
            "salary_min": {"type": "integer", "description": "Minimum salary in USD"},
            "salary_max": {"type": "integer", "description": "Maximum salary in USD"},
            "remote":     {"type": "boolean", "description": "Whether remote work is offered"}
        },
        "required": ["title", "company", "salary_min", "salary_max", "remote"]
    }
}

response = client.messages.create(
    model="claude-haiku-4-5",
    max_tokens=200,
    tools=[job_tool],
    tool_choice={"type": "tool", "name": "extract_job"},
    messages=[{"role": "user", "content": f"Extract the job info: {job_post}"}]
)

for block in response.content:
    if block.type == "tool_use":
        print("Job extraction:", block.input)
