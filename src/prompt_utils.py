"""
Utility functions for processing and formatting prompt data.
"""
import re
from typing import Dict, Any

def parse_question_data(question_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parses the raw question data into standardized components for prompt formatting.

    Args:
        question_data: The dictionary for a single question item
                       (expected keys: 'vignette', 'question', 'correctAnswer').

    Returns:
        A dictionary containing:
        'vignette': The vignette text (or default).
        'full_question': The original full question text (for reference).
        'question_stem': The part of the question before the options.
        'options_dict': A dict of options {'A': 'text', 'B': 'text', ...}.
        'options_text': A formatted string of options "A: text\\nB: text...".
        'correct_answer': The correct answer letter.
    """
    vignette = question_data.get('vignette', 'No vignette provided.')
    full_question = question_data.get('question', 'No question text provided.')
    correct_answer = question_data.get('correctAnswer')

    option_pattern = re.compile(
        r"^\s*(?:#\s*)?([A-D])[:\\.\\)]?\\s*(.*?)(?=\\n\\s*(?:#\\s*)?[A-D][: \\.\\)]|\\Z)", 
        re.MULTILINE | re.DOTALL | re.IGNORECASE
    )

    options_dict = {}
    options_text_parts = []
    question_stem = full_question 
    first_option_start_index = -1

    for match in option_pattern.finditer(full_question):
        option_letter = match.group(1).upper()
        option_text = re.sub(r'\\s+', ' ', match.group(2)).strip() 
        options_dict[option_letter] = option_text
        options_text_parts.append(f"{option_letter}: {option_text}")
        
        if first_option_start_index == -1:
            first_option_start_index = match.start()

    if first_option_start_index != -1:
        question_stem = full_question[:first_option_start_index].strip()
    
    options_text = "\\n".join(options_text_parts) if options_text_parts else "No options found."

    return {
        "vignette": vignette,
        "full_question": full_question,
        "question_stem": question_stem,
        "options_dict": options_dict,
        "options_text": options_text,
        "correct_answer": correct_answer
    } 