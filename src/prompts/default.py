"""
Stores default prompt templates.
"""

DEFAULT_PROMPT_TEMPLATE = """
Your task is to answer the following multiple-choice question.
**Your response MUST be ONLY the single letter of the correct option (A, B, or C).**
**Do NOT include any other text, reasoning, formatting, or explanation.**

Vignette:
{vignette}

Question Stem:
{question_stem}

Options:
A: {option_a}
B: {option_b}
C: {option_c}

Carefully read the vignette, question, and options. Choose the single best answer.

Answer (select one letter ONLY: A, B, or C):
"""
