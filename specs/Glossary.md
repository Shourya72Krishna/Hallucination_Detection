# Glossary — Terms used in Spec v1

**Token** — A small chunk of text (a word, part of a word, or 
punctuation) that the model breaks input/output into. The model 
generates one token at a time, in sequence. Used in the ERDP metric 
to mark "how far into the answer" detection happens.

**Layer** — One stage in the model's internal processing pipeline. 
Both Qwen2.5-1.5B and Llama-3.2-3B have 29 layers. Used in the ERDP 
metric to identify which layer detects hallucinations earliest.

**Hidden state** — The internal numeric representation of a token at 
a specific layer (1536 numbers for Qwen, 3072 for Llama). This is 
what the probe reads to make its prediction.

**Probe** — A small classifier that looks at a hidden state and 
predicts whether the content is true or a hallucination.

**ERDP (Earliest Reliable Detection Point)** — The earliest token 
position at which a probe's prediction becomes correct and stays 
correct for the rest of the generated answer. The core metric defined 
in Spec v1, Section 1.
