---
title: RAG vs Fine-tuning
tags: [GenAI, LLM, RAG]
---

RAG retrieves relevant context at query time and injects it into the prompt. No weight
changes — the base model stays frozen, and the knowledge lives in an external index
(e.g. a vector DB) that's cheap to update: add a document, re-embed, done.

Fine-tuning updates the model's weights permanently, baking new behavior or knowledge
directly into the parameters. It's better for teaching a model a new *style*, *format*,
or *skill* (e.g. following a specific output schema reliably) — things that are hard to
express as retrievable context. It's worse for facts that change often, since every
update requires a retraining pass.

Rule of thumb: RAG for *knowledge* that changes, fine-tuning for *behavior* that's stable.
They're not mutually exclusive — a fine-tuned model can still use RAG for facts.
