"""
Strategy implementation for the Self-Discover prompting technique.

This strategy guides the LLM to first generate a reasoning plan before answering.
"""
import logging
import time
import re 
import sys
import threading
from typing import List, Dict, Any
from ..llm_clients import get_llm_response
from ..prompts import self_discover as self_discover_prompt
from ..utils import ui_utils 
from ..utils.prompt_utils import parse_question_data 

logger = logging.getLogger(__name__)

def extract_final_answer_letter(text: str) -> tuple[str | None, bool]:
    """Extracts the final choice letter (A, B, C, D) from a response using multiple patterns."""    
    header_pattern = r"^(?:\\*\\*4?\\.\\s*Final Answer\\*\\*[:\\s]*|Final Answer:|The final answer is:|Based on \\w+ implementation, the correct option is)[\s\S]*?"
    answer_formats_after_header = [
        r"\\*\\*([A-D])\\*\\*[:\\.\\)]?",            
        r"\\b([A-D])[:\\.\\)]",                 
        r"letter\\s+([A-D])\\b",             
        r"is\\s+([A-D])\\b",                  
        r"option\\s+([A-D])\\b"               
    ]
    for ans_fmt in answer_formats_after_header:
        full_pattern = header_pattern + ans_fmt
        conclusion_match = re.search(full_pattern, text, re.IGNORECASE | re.MULTILINE)
        if conclusion_match:
            letter = conclusion_match.group(1).upper()
            logger.debug(f"Extracted letter '{letter}' using Pattern 1 ({ans_fmt=}) from Self-Discover: ...{text[-180:]}")
            return letter, True
    
    direct_choice_patterns = [
        r"^\\s*(?:-|\\*\\*|[•●])?\\s*\\*\\*([A-D])\\*\\*[:\\.\\)]?", 
        r"^\\s*(?:-|\\*\\*|[•●])?\\s*([A-D])[:\\.\\)]\\s*(?:is the (?:best|correct)|The (?:best|correct))", 
        r"^\\s*(?:-|\\*\\*|[•●])?\\s*\\*?([A-D])\\*?[:\\.\\)]", 
        r"^\\s*The (?:best|correct|final) (?:answer|option|choice) is\\s*\\*\\*([A-D])\\*\\*", 
        r"^\\s*The (?:best|correct|final) (?:answer|option|choice) is\\s+([A-D])\\b"    
    ]
    for pattern in direct_choice_patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            for group_val in match.groups(): 
                if group_val:
                    letter = group_val.upper()
                    logger.debug(f"Extracted letter '{letter}' using Pattern 2 ({pattern=}) from Self-Discover: ...{text[-180:]}")
                    return letter, True

    search_text_end = text[-300:]
    general_keyword_match_end = re.search( 
        r"""(?:Answer|Choice|Option) DNE\b""" + 
        r"""|(?:Therefore|Thus|So|Hence|Finally),?\s+(?:the|our|my)?\s*(?:final|correct|best)?\s*(?:answer|option|choice) is\s*(?:option|letter)?\s*\*\*([A-D])\*\*""" +
        r"""|(?:Therefore|Thus|So|Hence|Finally),?\s+(?:the|our|my)?\s*(?:final|correct|best)?\s*(?:answer|option|choice) is\s*(?:option|letter)?\s*\b([A-D])\b""" +
        r"""|(?:Final Answer|The answer is|Best choice is|The final choice is|Conclude by selecting|My choice is|The correct option is|Option is|Answer)[:\s]*[\s\S]*?\*\*([A-D])\*\*""" +
        r"""|(?:Final Answer|The answer is|Best choice is|The final choice is|Conclude by selecting|My choice is|The correct option is|Option is|Answer)[:\s]*[\s\S]*?\b([A-D])\b""" +
        r"""|\b([A-D])\.\s*\(?\w*\s*\w*\s*\w*\)?(?:\s+is correct|\s+is the answer)?[\.\s]?$""" + 
        r"""|\b([A-D])\s+is the correct answer\b""",
        search_text_end,
        re.IGNORECASE | re.MULTILINE
    )

    if general_keyword_match_end:
        for group in general_keyword_match_end.groups():
            if group:
                letter = group.upper()
                logger.debug(f"Extracted letter \'{letter}\' using Pattern 3 (general keywords, end of text) from Self-Discover: ...{search_text_end[-100:]}")
                return letter, True
    
    general_keyword_match_full = re.search(
        r"""(?:Answer|Choice|Option) DNE\b""" + 
        r"""|(?:Therefore|Thus|So|Hence|Finally),?\s+(?:the|our|my)?\s*(?:final|correct|best)?\s*(?:answer|option|choice) is\s*(?:option|letter)?\s*\*\*([A-D])\*\*""" +
        r"""|(?:Therefore|Thus|So|Hence|Finally),?\s+(?:the|our|my)?\s*(?:final|correct|best)?\s*(?:answer|option|choice) is\s*(?:option|letter)?\s*\b([A-D])\b""" +
        r"""|(?:Final Answer|The answer is|Best choice is|The final choice is|Conclude by selecting|My choice is|The correct option is|Option is|Answer)[:\s]*[\s\S]*?\*\*([A-D])\*\*""" +
        r"""|(?:Final Answer|The answer is|Best choice is|The final choice is|Conclude by selecting|My choice is|The correct option is|Option is|Answer)[:\s]*[\s\S]*?\b([A-D])\b""" +
        r"""|\b([A-D])\.\s*\(?\w*\s*\w*\s*\w*\)?(?:\s+is correct|\s+is the answer)?[\.\s]?""" + 
        r"""|\b([A-D])\s+is the correct answer\b""",
        text,
        re.IGNORECASE | re.MULTILINE
    )
    if general_keyword_match_full:
        for group in general_keyword_match_full.groups():
            if group:
                letter = group.upper()
                logger.debug(f"Extracted letter \'{letter}\' using Pattern 3 (general keywords, full text) from Self-Discover: ...{text[-100:]}")
                return letter, True
            
    end_of_text_match = re.search(
        r"""(?<![\w-])\b([A-D])\s*[\.\!]?\s*$""" + 
        r"""|(?:^\s*([A-D])\s*[\.\!]?\s*$)""",  
        text.strip(),
        re.IGNORECASE | re.MULTILINE
    )

    if end_of_text_match:
        for group in end_of_text_match.groups():
            if group:
                letter = group.upper()
                logger.debug(f"Extracted letter \'{letter}\' using Pattern 4 (end_of_text_match) from Self-Discover: ...{text[-100:]}")
                return letter, True
    logger.warning(f"Could not extract choice letter from Self-Discover response: ...{text[-180:]}")
    return None, False

def run_self_discover_strategy(mcq_data: List[Dict[str, Any]], model_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Runs the Self-Discover prompting strategy on a list of MCQs.

    Args:
        mcq_data: A list of dictionaries, each representing an MCQ.
        model_config: Configuration dictionary for the LLM API call.

    Returns:
        A list of dictionaries, each containing the original MCQ data
        plus the LLM's response, parsed answer, correctness, timing, etc.
    """
    results = []
    total_questions = len(mcq_data)
    config_id = model_config.get("config_id", model_config.get("model_id"))
    
    
    loading_animation = None
    for thread in threading.enumerate():
        if hasattr(thread, '_target') and thread._target and 'LoadingAnimation' in str(thread._target):
            frame = sys._current_frames().get(thread.ident)
            if frame:
                loading_animation = next((lv for lv in frame.f_locals.values() if isinstance(lv, ui_utils.LoadingAnimation)), None)
            if loading_animation: break
    
    if loading_animation: loading_animation.start()
    for i, question_item in enumerate(mcq_data):
        if loading_animation:
            loading_animation.update_progress(i + 1, total_questions)
            loading_animation.message = f"Processing Q{i+1}/{total_questions} with {config_id} (Self-Discover)"

        try:
            start_time = time.time() 
            prompt = self_discover_prompt.format_self_discover_prompt(question_item)
            model_type = model_config.get('type')            
            response_data = get_llm_response(prompt, model_config, is_json_response_expected=False) 
            
            if not response_data:
                 logger.error(f"API call failed for question {i+1}. response_data is None.")
                 
                 end_time = time.time()
                 results.append({
                     **question_item,
                     'LLM_response': "ERROR: API call failed (returned None)",
                     'LLM_answer': None,
                     'parse_successful': False,
                     'is_correct': None,
                     'response_time': end_time - start_time, 
                     'error': "API call failed"
                 })
                 continue  
            response_time = response_data.get('response_time')
            if response_time is None: 
                end_time = time.time()
                response_time = end_time - start_time
            llm_response_text = response_data.get('raw_response_text', 'ERROR: No raw response text found') 
            input_tokens = response_data.get('input_tokens')
            output_tokens = response_data.get('output_tokens')
            parsed_answer_letter, is_parse_successful = extract_final_answer_letter(llm_response_text)
            is_correct = None
            if is_parse_successful and 'correctAnswer' in question_item:
                is_correct = (str(parsed_answer_letter).strip().upper() == 
                              str(question_item['correctAnswer']).strip().upper())
            elif 'correctAnswer' not in question_item:
                 logger.warning(f"Question {i+1} missing 'correctAnswer'. Cannot determine correctness.")
            elif not is_parse_successful:
                 logger.warning(f"Failed to parse answer for question {i+1}. Response snippet: {llm_response_text[-100:]}")
                 is_correct = False 
            results.append({
                **question_item,
                'LLM_response': llm_response_text,
                'LLM_answer': parsed_answer_letter,
                'parse_successful': is_parse_successful,
                'is_correct': is_correct,
                'response_time': response_time,
                'input_tokens': input_tokens,
                'output_tokens': output_tokens,
                'answer_length': len(parsed_answer_letter) if parsed_answer_letter else 0
            })
        except Exception as e:            
            end_time = time.time() 
            elapsed_time = end_time - start_time if 'start_time' in locals() else 0
            logger.error(f"Error processing question {i+1} with Self-Discover strategy: {e}", exc_info=True)
            results.append({
                **question_item,
                'LLM_response': f"ERROR: {e}",
                'LLM_answer': None,
                'parse_successful': False,
                'is_correct': None,
                'response_time': elapsed_time, 
                'error': str(e)
            })

    if loading_animation: loading_animation.stop()
    logger.info(f"Self-Discover strategy completed for {len(results)} items using {config_id}.")
    return results 