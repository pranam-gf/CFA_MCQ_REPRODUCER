"""
Stores Chain-of-Thought (CoT) related prompt templates.
"""

COHERENT_CFA_COT = """
You are a Chartered Financial Analyst (CFA) charterholder.  Your task is to answer one multiple‐choice question from the CFA curriculum.  Follow these steps:

1. Restate the question in your own words.
2. Think through it step by step, showing your reasoning (use bullet points if helpful).
3. Evaluate each of the four choices (A, B, C, D), noting why each could be right or wrong.
4. Conclude by selecting the single best answer (just the letter and a one-sentence justification).

Vignette:
{vignette}

Question Stem:
{question_stem}

A) {opt_a}
B) {opt_b}
C) {opt_c}
D) {opt_d}

text

Answer:
"""

SELF_CONSISTENCY_INSTRUCTIONS = """
Repeat the above chain-of-thought prompt N times (with temperature > 0) to get multiple independent reasoning traces.  
Finally, perform a majority vote on the selected letters to pick your final answer.
""" 