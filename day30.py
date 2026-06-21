# Day 30 — LLMs & How They Work
# Week 6 Day 5

import os
import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as F
from dotenv import load_dotenv
from anthropic import Anthropic

# ============================================================
# TOPIC 1: What an LLM Is
# ============================================================
# An LLM is a next-token classifier: given a sequence of tokens, it outputs
# a softmax probability distribution over the ENTIRE vocabulary (~100,000
# tokens). Generation = predict one token, append it, repeat (autoregressive).
# Same softmax + CrossEntropyLoss mechanics as Day 28's 10-digit MNIST
# classifier — just nn.Linear(in_features, 100_000) instead of (32, 10).

# ============================================================
# TOPIC 2: Tokens — How Text Becomes Numbers
# ============================================================
# Subword tokenization can represent ANY string, even words it's never seen
# as a whole unit — it just falls back to smaller pieces. Rare/unusual text
# costs more tokens (and more money — APIs bill per token, not per word).

encoding = tiktoken.get_encoding("cl100k_base")

text = "Supercalifraglisticexpialidocious"
tokens = encoding.encode(text)

print(f"Token IDs: {tokens}")
print(f"Token count: {len(tokens)} | Character count: {len(text)}")
for token_id in tokens:
    print(f"  {token_id} -> {repr(encoding.decode([token_id]))}")

# ============================================================
# TOPIC 3: Embeddings — Token IDs to Vectors
# ============================================================
# A token ID is an arbitrary integer with no inherent meaning. nn.Embedding
# is a learned lookup table mapping each ID to a dense vector — tokens used
# in similar contexts end up with similar vectors after training. Random
# (meaningless) at init, same as Day 28's untrained conv filters.

vocab_size = 100_000      # matches cl100k_base's vocabulary size
embedding_dim = 8          # tiny for illustration — real models use 768 to 12,000+

embedding_layer = nn.Embedding(vocab_size, embedding_dim)
token_ids = torch.tensor([464, 2415, 3332])    # 3 example token IDs
vectors = embedding_layer(token_ids)
print(f"\nEmbedding shape: {vectors.shape}")   # (3, 8) — each ID -> 8-dim vector

# ============================================================
# TOPIC 4: The Transformer Architecture
# ============================================================
# CNNs (Day 28) only see nearby pixels via small kernels — too local for
# text, where relevant relationships can span any distance ("it" referring
# to a noun many words earlier). Self-attention (Topic 5) lets every token
# connect directly to every other token in one step, regardless of distance.
# A Transformer block is mostly familiar parts: embedding -> self-attention
# -> nn.Linear + activation (Day 26/27) -> repeated N times -> softmax output.

# ============================================================
# TOPIC 5: Self-Attention — Just Matmul + Softmax
# ============================================================
# Q (what am I looking for) / K (what do I contain) / V (what I offer) come
# from learned weight matrices. Score = Q @ K.T (similarity), softmax turns
# scores into weights summing to 1, output = weights @ V (weighted blend of
# every token's Value). Entirely matrix multiplication + softmax — nothing
# mathematically new versus Day 26's forward pass.

X = torch.tensor([[1.0, 0.0, 1.0, 0.0],   # token 1: "The"
                   [0.0, 1.0, 0.0, 1.0],   # token 2: "trophy"
                   [1.0, 1.0, 0.0, 0.0]])  # token 3: "it"

Q, K, V = X, X, X
scores  = Q @ K.T                     # (3,3) — similarity between every pair of tokens
weights = F.softmax(scores, dim=-1)   # softmax each row into a probability distribution
attention_output = weights @ V        # weighted sum of Value vectors

print(f"\nAttention weights:\n{weights}")
print(f"Attention output:\n{attention_output}")

# ============================================================
# TOPIC 6: Training Objective — Next-Token Prediction
# ============================================================
# The label for "predict the next token" is just the next word in the raw
# text itself — no human labeling needed (self-supervised learning). This is
# why LLMs can train on practically the whole internet for free, unlike
# Day 27's Titanic model (needed a human-labeled Survived column) or Day 28's
# MNIST (needed human-labeled digit images).

# ============================================================
# TOPIC 7: Pretraining -> Fine-Tuning -> RLHF
# ============================================================
# Pretraining (next-token prediction on internet-scale text) produces a base
# model that just continues text plausibly — no concept of "answer the
# question." Supervised fine-tuning on curated (instruction, response) pairs
# teaches the assistant format. RLHF uses human-ranked comparisons to further
# align outputs with what people actually find helpful/safe/honest. Same
# gradient descent training loop at every stage — only the data changes.

# ============================================================
# TOPIC 8: First Real LLM API Call
# ============================================================
# client.messages.create() is a POST request under the hood — same
# request/response pattern as Day 29's Flask API, just calling Anthropic's
# server instead of our own. response.usage reports the actual token counts
# (Topic 2) that get billed.

load_dotenv()   # reads .env into environment variables — key never appears in this file

client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

response = client.messages.create(
    model="claude-haiku-4-5",
    max_tokens=200,
    system="You are a concise, friendly Python tutor.",
    messages=[
        {"role": "user", "content": "In one paragraph, explain what a token is to someone learning AI engineering."}
    ]
)

print(f"\nClaude's response:\n{response.content[0].text}")
print(f"\nTokens used — input: {response.usage.input_tokens}, output: {response.usage.output_tokens}")
