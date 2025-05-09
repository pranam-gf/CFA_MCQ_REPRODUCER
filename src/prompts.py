"""
Stores prompt templates for different LLM types.
"""

GEMINI_PROMPT_TEMPLATE = """Given the following context (vignette) and a multiple-choice question (which includes the question stem and all options), please analyze the information and select the best answer.

Respond with ONLY the single uppercase letter of your chosen option (e.g., A, B, C). Do not include any other text, explanation, or punctuation.

Context (Vignette):
{vignette}

Question and Options:
{question_full_text}

Your chosen option letter:"""

DEFAULT_PROMPT_TEMPLATE = """
Consider the following vignette:
{vignette}

Based on the vignette, answer the following question:
{question_full_text}

Respond with ONLY the single letter corresponding to the correct answer (e.g., A, B, or C). Do not include any other text, explanation, or formatting.
""" 