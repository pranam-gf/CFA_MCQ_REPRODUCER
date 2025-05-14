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
        'options_text': A formatted string of options "A: text\nB: text...".
        'correct_answer': The correct answer letter.
    """
    vignette = question_data.get('vignette', 'No vignette provided.')
    full_question = question_data.get('question', 'No question text provided.')
    correct_answer = question_data.get('correctAnswer')
    options_dict = {}
    if 'option_a' in question_data:
        options_dict['A'] = question_data['option_a']
    if 'option_b' in question_data:
        options_dict['B'] = question_data['option_b']
    if 'option_c' in question_data:
        options_dict['C'] = question_data['option_c']
    
    if not options_dict:
        option_pattern = re.compile(
            r"^\s*(?:#\s*)?([A-D])[:\.\)]?\s*(.*?)(?=\n\s*(?:#\s*)?[A-D][: \.\)]|\Z)", 
            re.MULTILINE | re.DOTALL | re.IGNORECASE
        )

        first_option_start_index = -1

        for match in option_pattern.finditer(full_question):
            option_letter = match.group(1).upper()
            option_text = re.sub(r'\s+', ' ', match.group(2)).strip() 
            options_dict[option_letter] = option_text
            
            if first_option_start_index == -1:
                first_option_start_index = match.start()

        if first_option_start_index != -1:
            question_stem = full_question[:first_option_start_index].strip()
        else:
            question_stem = full_question
    else:
        question_stem = full_question
    options_text_parts = []
    for letter in sorted(options_dict.keys()):
        options_text_parts.append(f"{letter}: {options_dict[letter]}")
    
    options_text = "\n".join(options_text_parts) if options_text_parts else "No options found."

    return {
        "vignette": vignette,
        "full_question": full_question,
        "question_stem": question_stem,
        "options_dict": options_dict,
        "options_text": options_text,
        "correct_answer": correct_answer
    } 