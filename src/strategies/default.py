"""
Default processing strategy: process each question once.
"""
import logging
import sys
import threading
import re 
from .. import llm_clients 
from ..utils import ui_utils
from ..utils.prompt_utils import parse_question_data
from ..prompts import default as default_prompts

logger = logging.getLogger(__name__)

def generate_prompt_for_default_strategy(entry: dict, model_type: str | None = None) -> str:
    """Generates the default prompt for the LLM using the standardized parser."""
    parsed_data = parse_question_data(entry)
    
    template = default_prompts.DEFAULT_PROMPT_TEMPLATE
    if model_type == "gemini":
        template = default_prompts.GEMINI_PROMPT_TEMPLATE
        
    return template.format(
        vignette=parsed_data['vignette'],
        question_stem=parsed_data['question_stem'],
        options_text=parsed_data['options_text']
    )

def run_default_strategy(data: list[dict], model_config_item: dict) -> list[dict]:
    """
    Processes each question entry using the specified LLM, expecting only a single letter answer.
    Uses the standardized prompt generation.
    """
    results = []
    total_questions = len(data)
    config_id = model_config_item.get("config_id", model_config_item.get("model_id"))
    model_type = model_config_item.get("type")
    
    loading_animation = None
    
    for thread in threading.enumerate():
        if hasattr(thread, '_target') and thread._target and 'LoadingAnimation' in str(thread._target):
            frame = sys._current_frames().get(thread.ident)
            if frame:
                loading_animation = next((lv for lv in frame.f_locals.values() if isinstance(lv, ui_utils.LoadingAnimation)), None)
            if loading_animation: 
                break

    for i, entry in enumerate(data):
        if loading_animation:
            loading_animation.update_progress(i + 1, total_questions)
            loading_animation.message = f"Processing Q{i+1}/{total_questions} with {config_id} (Default Strategy)"

        logger.info(f"Processing question {i + 1}/{total_questions} with model {config_id} (Default Strategy)...")
        
        prompt = generate_prompt_for_default_strategy(entry, model_type=model_type)
        
        llm_data = llm_clients.get_llm_response(prompt, model_config_item)

        llm_answer = ""
        is_correct = None
        response_time = None
        current_input_tokens = None
        current_output_tokens = None
        answer_length = 0
        raw_response_text = None
        error_message = None

        if llm_data:
            response_time = llm_data.get('response_time')
            current_input_tokens = llm_data.get('input_tokens')
            current_output_tokens = llm_data.get('output_tokens')
            raw_response_text = llm_data.get('raw_response_text')
            error_message = llm_data.get('error_message')

            if raw_response_text:
                
                
                
                matches = re.findall(r"\b([A-D])\b", raw_response_text, re.IGNORECASE)
                if matches:
                    
                    llm_answer = matches[-1].upper()
                    logger.info(f"Q {i + 1} ({config_id}): Default strategy extracted last standalone letter '{llm_answer}' from response: '{raw_response_text[:100]}...'")
                else:
                    
                    logger.warning(f"Q {i + 1} ({config_id}): Default strategy failed to extract any standalone A, B, C, or D from response: '{raw_response_text[:100]}...'")
                    llm_answer = "PARSE_FAIL"
            else: 
                 logger.warning(f"Q {i + 1} ({config_id}): Default strategy received no raw response text.")
                 llm_answer = "PARSE_FAIL" 

            
            if error_message:
                logger.warning(f"Q {i + 1} ({config_id}): Problem with LLM call: {error_message}")
                llm_answer = "ERROR"
            
            answer_length = len(llm_answer) if llm_answer not in ["ERROR", "PARSE_FAIL"] else 0

            
            correct_answer_str = str(entry.get('correctAnswer', '')).strip().upper()
            if not correct_answer_str or "PLACEHOLDER" in correct_answer_str:
                logger.warning(f"Q {i + 1} ({config_id}): Correct answer missing/placeholder. Cannot evaluate correctness.")
                is_correct = None
            elif llm_answer in ["ERROR", "PARSE_FAIL"]:
                is_correct = False
                logger.warning(f"Q {i + 1} ({config_id}): LLM response resulted in error or parse failure ('{llm_answer}'). Marked as incorrect.")
            elif not llm_answer or llm_answer not in ["A", "B", "C", "D"]: 
                is_correct = False
                logger.warning(f"Q {i + 1} ({config_id}): LLM provided an empty or invalid answer ('{llm_answer}'). Marked as incorrect.")
            else:
                is_correct = (llm_answer == correct_answer_str)

            log_time = f", Time: {response_time:.2f}s" if response_time is not None else ""
            logger.info(f"Q {i + 1} ({config_id}): LLM Ans: '{llm_answer}', Correct: '{correct_answer_str}', Match: {is_correct}{log_time}")
        else:
            logger.error(f"Q {i + 1} ({config_id}): Failed to get any valid LLM response (API call likely failed).")
            llm_answer = "ERROR"
            is_correct = False

        updated_entry = entry.copy()
        updated_entry.update({
            'LLM_answer': llm_answer,
            'is_correct': is_correct,
            'response_time': response_time,
            'input_tokens': current_input_tokens,
            'output_tokens': current_output_tokens,
            'answer_length': answer_length,
            'raw_response_text': raw_response_text,
            'error': error_message,
            'prompt_strategy': 'default'
        })
        results.append(updated_entry)

    if loading_animation: 
        loading_animation.stop()
        
    return results 