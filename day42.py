# Day 42 — Evaluating AI Agents
# Week 9 Day 3
#
# TOPIC 2: Deterministic evals check outputs using predetermined rules — keywords
# that must appear, exact values, minimum match counts. Your code controls all
# scoring logic; Claude just produces the answer. Reliable for questions with
# exact answers (names, numbers, dates) but brittle for open-ended responses.
#
# TOPIC 3: LLM-as-judge passes the question, the answer, and an ideal answer to
# a second Claude call which scores 1-5. More flexible than keyword matching
# because it understands context — but the judge shares the same model biases
# as the agent. Use both together: deterministic for facts, judge for quality.

import os
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()
client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

# ============================================================
# TOPIC 2: Deterministic evals — keyword and exact match
# ============================================================

TEST_CASES = [
    {
        "description": "Capital city lookup",
        "input": "What is the capital of France?",
        "expected_keywords": ["Paris"],
        "min_matches": 1
    },
    {
        "description": "Basic arithmetic",
        "input": "What is 15 multiplied by 23?",
        "expected_keywords": ["345"],
        "min_matches": 1
    },
    {
        "description": "Multiple keyword match",
        "input": "Name two programming languages",
        "expected_keywords": ["Python", "JavaScript", "Java", "C++", "Ruby"],
        "min_matches": 2
    },
]

def run_agent(user_input):
    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=200,
        messages=[{"role": "user", "content": user_input}]
    )
    return response.content[0].text

def evaluate(test_case):
    answer = run_agent(test_case["input"])
    keywords = test_case["expected_keywords"]
    min_matches = test_case["min_matches"]
    matches = [k for k in keywords if k.lower() in answer.lower()]
    passed = len(matches) >= min_matches
    return passed, matches, answer

def run_all_evals():
    results = []
    for tc in TEST_CASES:
        passed, matches, answer = evaluate(tc)
        results.append(passed)
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {tc['description']}")
        if not passed:
            print(f"       Expected: {tc['expected_keywords']}")
            print(f"       Got: {answer[:100]}")

    score = sum(results)
    print(f"\nScore: {score}/{len(TEST_CASES)}")

# ============================================================
# TOPIC 3: LLM-as-judge — Claude grades Claude's outputs
# ============================================================

JUDGE_CASES = [
    {
        "description": "Capital city question",
        "input": "What is the capital of France?",
        "ideal": "Paris"
    },
    {
        "description": "Explanation quality",
        "input": "Explain what an API is in one sentence.",
        "ideal": "A way for two programs to communicate with each other"
    },
    {
        "description": "Factual accuracy",
        "input": "Who invented the telephone?",
        "ideal": "Alexander Graham Bell"
    },
]

def judge(question, answer, ideal):
    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=100,
        system="You are a strict grader. Score the answer 1-5 based on correctness and relevance to the ideal answer. Reply with just a number.",
        messages=[{
            "role": "user",
            "content": f"Question: {question}\nIdeal answer: {ideal}\nActual answer: {answer}\nScore:"
        }]
    )
    return int(response.content[0].text.strip())

def run_judge_evals():
    total = 0
    for tc in JUDGE_CASES:
        answer = run_agent(tc["input"])
        score = judge(tc["input"], answer, tc["ideal"])
        total += score
        print(f"[{score}/5] {tc['description']}")
        print(f"       Answer: {answer[:80]}")

    avg = total / len(JUDGE_CASES)
    print(f"\nAverage score: {avg:.1f}/5")

if __name__ == "__main__":
    run_all_evals()
    print()
    run_judge_evals()



    
