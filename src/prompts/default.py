"""
Stores default prompt templates.
"""

GEMINI_PROMPT_TEMPLATE = """
I need you to read a vignette and multiple-choice question from the CFA exam and select the correct answer.

Instructions:
1. Read the vignette and question text carefully.
2. Analyze the problem and determine the single best answer from the options provided.
3. Your response will be parsed programmatically. It is ESSENTIAL that you follow the output format precisely.

Vignette:
{vignette}

Question Stem:
{question_stem}

Options:
A: {option_a}
B: {option_b}
C: {option_c}

Choose the single best answer.
Your final output MUST be a single uppercase letter (A, B, C, or D) and NOTHING ELSE. Any deviation will result in a failure.
Answer:"""

DEFAULT_PROMPT_TEMPLATE = """
Your task is to answer the following multiple-choice question.
**Your response MUST be ONLY the single letter of the correct option (A, B, C, or D).**
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

Answer (select one letter ONLY: A, B, C, or D):
"""
