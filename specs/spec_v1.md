# Spec v1 — Core Methodology
Week 1

## 0. Context (plain-language summary)

We're testing whether an LLM's internal "hidden states" (its internal 
numeric representation of each word it generates, at every layer) contain 
a signal that predicts whether the answer being generated is true or a 
hallucination — and how early we can detect that signal, before the 
model finishes its full answer.

This spec defines three things everyone downstream needs:
1. Exactly how we measure "how early" (the metric)
2. Exactly how we split each dataset for training/testing
3. Exactly how each dataset gets labelled true/false

---

## 1. Earliest-Reliable-Detection Metric

**Setup:** For one generated answer with T tokens, evaluated at every 
layer l (1 to 29) and every token position t (1 to T), we have:

- `probe_pred(t, l)` — the probe's guess (true / hallucination) using 
  the hidden state at token t, layer l, valid only when the probe's 
  confidence ≥ θ (θ = 0.7 to start; tuned later in the Week 5–6 sweep)
- `ground_truth` — the correct label for the full answer, from the 
  dataset (see Section 3)

**Definition — Earliest Reliable Detection Point (ERDP):**

For a fixed layer l:

    ERDP(l) = the smallest token position t such that probe_pred(t', l) 
              equals ground_truth for every t' from t onward to T

In plain words: the earliest point in the answer after which the probe 
never flips to a wrong guess again. A single lucky correct guess that 
then flips back to wrong does NOT count — it has to stay correct.

**Per-example reporting:**
- Best layer: `l* = argmin_l ERDP(l)` (whichever layer detects earliest)
- `ERDP(l*)` — the raw token position
- `ERDP(l*) / T` — normalized position (0 = start of answer, 1 = end), 
  so answers of different lengths are comparable

**Per-dataset/model reporting (aggregated, per Seed Protocol, Week 4):**
- Mean ± standard deviation of normalized ERDP across 5 random seeds
- Failure rate: % of examples where the probe never becomes reliable 
  within the full T tokens

This aggregated curve (detection point vs. reliability) is the paper's 
central figure — the "earliest-reliable-detection curve."

---

## 2. Train / Validation / Test Split Strategy

**Per-dataset split (applied independently to each of the three datasets):**
- 70% train / 15% validation / 15% test
- Stratified by label (true/hallucination) so each split has a similar 
  proportion of both classes
- Fixed random seed for the split itself, so it's reproducible

**Why this matters for later weeks (cross-dataset generalisation):**
Each dataset's TEST split must remain fully held out and untouched by 
training on that same dataset. This lets us later (Week 6–7) train a 
probe only on Dataset A's train split, and test it cold on Dataset B's 
and Dataset C's test splits — with no overlap, no leakage, no need to 
redo any split.

**Practical instruction for Person 2 (Week 2):**
When building `labeled_dataset_v1`, tag every example with three columns: 
`dataset_name`, `split` (train/val/test), and `label`. This one flat 
structure is what every later step (extraction, probing, generalisation 
matrix) reads from.

---

## 3. Labelling Rule

**TruthfulQA and HaluEval:**
Use the existing human-annotated labels exactly as published. No 
relabelling, no modification.

**SimpleQA (no human labels exist — must auto-label):**

1. **Exact-match check:** Compare the generated answer against the 
   dataset's reference answer, after normalizing (lowercase, strip 
   punctuation/whitespace). Match → label = TRUE.

2. **Semantic-similarity check (if exact match fails):** Encode both 
   answers using a sentence-embedding model (e.g., `all-MiniLM-L6-v2`). 
   Compute cosine similarity.
   - Similarity ≥ 0.85 → label = TRUE
   - Similarity < 0.85 → label = HALLUCINATION

3. **Edge case handling:** Empty answer, refusal ("I don't know"), or 
   off-topic → label = HALLUCINATION by default.

Person 2 should sample ~30 borderline cases manually in Week 2 to 
sanity-check the 0.85 threshold before running at full scale.

---

## 4. Base Models (finalised)

- Qwen/Qwen2.5-1.5B — commit `8faed76` — 29 layers, hidden dim 1536
- meta-llama/Llama-3.2-3B — commit `13afe51` — license accepted — 
  29 layers, hidden dim 3072
