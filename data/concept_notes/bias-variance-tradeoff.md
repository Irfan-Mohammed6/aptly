---
title: Bias-Variance Tradeoff
tags: [ML fundamentals, model evaluation]
---

Bias is error from a model that's too simple to capture the underlying pattern
(underfitting) — it makes similarly-wrong predictions regardless of which training set
it saw. Variance is error from a model that's too sensitive to the specific training set
it happened to see (overfitting) — it makes wildly different predictions if you retrain
it on a slightly different sample.

Total expected error decomposes roughly as bias^2 + variance + irreducible noise. Adding
model capacity (more parameters, more depth, fewer regularization constraints) typically
trades bias for variance: training error drops (less bias) but the gap between training
and validation error grows (more variance). The practical signal to watch: a large
train/validation gap points at high variance (overfitting — regularize, get more data,
simplify); a small gap but a high absolute error on both points at high bias
(underfitting — add capacity, better features, less regularization).
