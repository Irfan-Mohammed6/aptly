---
title: Transformer Self-Attention
tags: [GenAI, transformers, architecture]
---

Self-attention lets each token in a sequence look at every other token and weigh how
relevant each one is when building its own updated representation. Concretely: every
token is projected into a Query, Key, and Value vector; a token's Query is dot-producted
against every other token's Key to get relevance scores, those scores are softmaxed into
weights, and the output is the weighted sum of all Value vectors.

This is what lets a model resolve things like pronoun references ("it" attending back to
the noun it refers to) without needing a hand-built rule — the attention weights are
learned. Multi-head attention just runs several of these in parallel with different
learned projections, so different heads can specialize in different kinds of
relationships (syntax, coreference, etc.), and their outputs get concatenated.

Why it replaced RNNs for this: attention computes all pairwise relationships in one
parallelizable step, whereas an RNN has to process tokens sequentially — attention
trades O(n) sequential steps for O(n^2) attention scores, which is a good trade on
modern parallel hardware for the sequence lengths in practice.
