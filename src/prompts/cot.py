"""
Stores Chain-of-Thought (CoT) related prompt templates.
"""

COHERENT_CFA_COT = """
You are a Chartered Financial Analyst (CFA) charterholder.  Your task is to answer one multiple‐choice question from the CFA curriculum.  Follow these steps:

1. Restate the question stem in your own words.
2. Think through it step by step, showing your reasoning (use bullet points if helpful).
3. Evaluate each of the choices provided in the Options section, noting why each could be right or wrong.
4. Conclude by selecting the single best answer. First, provide a one-sentence justification for your choice. Then, on a new, separate line, write "Final Answer: [LETTER]", where [LETTER] is the capital letter of your chosen option (e.g., "Final Answer: A"). This "Final Answer: [LETTER]" line must be the absolute last line of your response.

Vignette:
{vignette}

Question Stem:
{question_stem}

Options:
A: {option_a}
B: {option_b}
C: {option_c}

Answer:
"""

SELF_CONSISTENCY_INSTRUCTIONS = """
Repeat the above chain-of-thought prompt N times (with temperature > 0) to get multiple independent reasoning traces.  
Finally, perform a majority vote on the selected letters to pick your final answer.
""" 