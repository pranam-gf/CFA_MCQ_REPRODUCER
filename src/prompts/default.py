"""
Stores default prompt templates.
"""

GEMINI_PROMPT_TEMPLATE = """
I need you to read a vignette and multiple-choice question from the CFA exam and select the correct answer.

Instructions:
1. Read the vignette and question text carefully
2. Choose the single best answer from options A, B, C, or D
3. Reply with ONLY the letter of your answer (A, B, C, or D)
4. Do not include explanations or reasoning

Vignette:
{vignette}

Question:
{question_full_text}

Your response should be a single letter: A, B, C, or D
"""

DEFAULT_PROMPT_TEMPLATE = """
Read the following vignette and multiple-choice question carefully.
Your task is to select the single best answer from the options provided (A, B, C, or D).
Do not provide any explanation or reasoning, only the letter of your chosen answer.

Vignette:
{vignette}

Question:
{question_full_text}

Your Answer (select one: A, B, C, or D):
"""
