# Day 44 — Week 9 Capstone: Customer Review Pipeline
# Week 9 Day 5
#
# Combines all four Week 9 topics into one production pipeline:
# TOPIC 1: Stage 1 (classify) uses few-shot prompting (Day 43) to label each
# review POSITIVE/NEGATIVE/MIXED. Stage 2 (extract) uses tool forcing (Day 41)
# to pull out sentiment, score, key_issue, and needs_response as a clean dict.
#
# TOPIC 2: Stage 3 (reply) streams a draft customer response (Day 40) only when
# needs_response is True — the structured output from Stage 2 controls the flow.
#
# TOPIC 3: Eval suite (Day 42) runs the full pipeline on test cases and checks
# both the category and needs_response against expected values, reporting PASS/FAIL.

import os
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()
client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

# ============================================================
# TOPIC 1: Classify + Extract — stages 1 and 2 of the pipeline
# ============================================================

CLASSIFY_SYSTEM = """Classify the customer review as POSITIVE, NEGATIVE, or MIXED.

Examples:
Review: "Amazing product, works perfectly!"
Category: POSITIVE

Review: "Terrible experience, broke after one day"
Category: NEGATIVE

Review: "Good features but too expensive"
Category: MIXED

Reply with just the category word. Nothing else."""

def classify_review(review):
    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=10,
        system=CLASSIFY_SYSTEM,
        messages=[{"role": "user", "content": review}]
    )
    return response.content[0].text.strip()

extract_tool = {
    "name": "extract_review",
    "description": "Extract structured data from a customer review.",
    "input_schema": {
        "type": "object",
        "properties": {
            "sentiment":      {"type": "string",  "description": "positive, negative, or mixed"},
            "score":          {"type": "integer", "description": "1-10 satisfaction score"},
            "key_issue":      {"type": "string",  "description": "Main point of the review in one sentence"},
            "needs_response": {"type": "boolean", "description": "True if the review requires a business response"}
        },
        "required": ["sentiment", "score", "key_issue", "needs_response"]
    }
}

def extract_review(review):
    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=200,
        tools=[extract_tool],
        tool_choice={"type": "tool", "name": "extract_review"},
        messages=[{"role": "user", "content": f"Extract data from this review: {review}"}]
    )
    for block in response.content:
        if block.type == "tool_use":
            return block.input

# if __name__ == "__main__":
#     reviews = [
#         "I've been using this app for 3 months and it's completely transformed my workflow. 10/10!",
#         "The product arrived broken and customer service never replied to my emails. Absolute disaster.",
#         "Decent product overall but the setup process was way too complicated for new users.",
#     ]

#     for review in reviews:
#         category = classify_review(review)
#         data = extract_review(review)
#         print(f"Review: {review[:60]}...")
#         print(f"Category: {category}")
#         print(f"Data: {data}")
#         print()

# ============================================================
# TOPIC 2: Stream a draft reply — stage 3 of the pipeline
# ============================================================

import time

REPLY_SYSTEM = """You are a customer service agent writing replies to reviews.

Rules:
- Be empathetic and professional
- Keep it to 2-3 sentences max
- For negative reviews: apologise and offer to help
- For positive reviews: thank them warmly
- Never make promises about refunds or replacements"""

def stream_reply(review, sentiment):
    print("Draft reply: ", end="", flush=True)
    with client.messages.stream(
        model="claude-haiku-4-5",
        max_tokens=150,
        system=REPLY_SYSTEM,
        messages=[{"role": "user", "content": f"Write a reply to this {sentiment} review: {review}"}]
    ) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)
            time.sleep(0.01)
    print()

# if __name__ == "__main__":
#     reviews = [
#         "I've been using this app for 3 months and it's completely transformed my workflow. 10/10!",
#         "The product arrived broken and customer service never replied to my emails. Absolute disaster.",
#         "Decent product overall but the setup process was way too complicated for new users.",
#     ]

#     for review in reviews:
#         category = classify_review(review)
#         data = extract_review(review)
#         print(f"Review: {review[:60]}...")
#         print(f"Category: {category} | Score: {data['score']}/10 | Needs response: {data['needs_response']}")
#         if data["needs_response"]:
#             stream_reply(review, data["sentiment"])
#         print()

# ============================================================
# TOPIC 3: Eval suite — testing the full pipeline
# ============================================================

EVAL_CASES = [
    {
        "review": "Absolutely love this product, best purchase I've made all year!",
        "expected_category": "POSITIVE",
        "expected_needs_response": False
    },
    {
        "review": "This is a scam, I want my money back immediately",
        "expected_category": "NEGATIVE",
        "expected_needs_response": True
    },
    {
        "review": "Works well for basic tasks but lacks advanced features",
        "expected_category": "MIXED",
        "expected_needs_response": True
    },
]

def run_pipeline_evals():
    passed = 0

    for tc in EVAL_CASES:
        review = tc["review"]
        category = classify_review(review)
        data = extract_review(review)

        cat_pass  = category == tc["expected_category"]
        resp_pass = data["needs_response"] == tc["expected_needs_response"]

        if cat_pass and resp_pass:
            passed += 1
            print(f"[PASS] {review[:50]}...")
        else:
            print(f"[FAIL] {review[:50]}...")
            if not cat_pass:
                print(f"       Category: expected {tc['expected_category']}, got {category}")
            if not resp_pass:
                print(f"       needs_response: expected {tc['expected_needs_response']}, got {data['needs_response']}")

    print(f"\nPipeline score: {passed}/{len(EVAL_CASES)}")

if __name__ == "__main__":
    run_pipeline_evals()