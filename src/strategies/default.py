"""
Default processing strategy: process each question once.
"""
import logging
import sys
import threading
import re 
from .. import llm_clients 
from .. import ui_utils 

logger = logging.getLogger(__name__)

def generate_prompt_for_default_strategy(entry: dict, model_type: str | None = None, prompt_type: str | None = "default") -> str:
    """Generates the prompt for the LLM using specified templates and structured data fields."""
    from ..prompts import default as default_prompts

    vignette = entry.get('vignette', 'Vignette not available.')
    question_stem = entry.get('question', 'Question stem not available.')
    
    opt_a_text = entry.get('option_a', '')
    opt_b_text = entry.get('option_b', '')
    opt_c_text = entry.get('option_c', '')
    opt_d_text = entry.get('option_d', '') 

    if prompt_type == "COHERENT_CFA_COT":
        actual_opt_d_for_prompt = opt_d_text if opt_d_text else "N/A (Not applicable)"

        if not hasattr(default_prompts, 'COHERENT_CFA_COT_PROMPT_TEMPLATE'):
            logger.error("COHERENT_CFA_COT_PROMPT_TEMPLATE not found in prompts.default. Using fallback string.")
            
            return f"""Vignette: {vignette}
Question Stem: {question_stem}
Option A: {opt_a_text}
Option B: {opt_b_text}
Option C: {opt_c_text}
Option D: {actual_opt_d_for_prompt}
(Error: Prompt template missing)"""

        return default_prompts.COHERENT_CFA_COT_PROMPT_TEMPLATE.format(
            vignette=vignette, 
            parsed_stem=question_stem, 
            opt_a=opt_a_text or "N/A", 
            opt_b=opt_b_text or "N/A",
            opt_c=opt_c_text or "N/A",
            opt_d=actual_opt_d_for_prompt
        )
    else:
        
        options_parts = []
        if opt_a_text:
            options_parts.append(f"A) {opt_a_text}")
        if opt_b_text:
            options_parts.append(f"B) {opt_b_text}")
        if opt_c_text:
            options_parts.append(f"C) {opt_c_text}")
        if opt_d_text: 
            options_parts.append(f"D) {opt_d_text}")
        
        options_string = "\n".join(options_parts)
        question_full_text = f"{question_stem}\n\n{options_string}".strip()

        if model_type == "gemini":
            if not hasattr(default_prompts, 'GEMINI_PROMPT_TEMPLATE'):
                 logger.error("GEMINI_PROMPT_TEMPLATE not found in prompts.default. Using fallback.")
                 return f"""Vignette: {vignette}\nQuestion: {question_full_text}\n(Error: Prompt template missing)"""
            return default_prompts.GEMINI_PROMPT_TEMPLATE.format(vignette=vignette, question_full_text=question_full_text)
        else: 
            if not hasattr(default_prompts, 'DEFAULT_PROMPT_TEMPLATE'):
                logger.error("DEFAULT_PROMPT_TEMPLATE not found in prompts.default. Using fallback.")
                return f"""Vignette: {vignette}\nQuestion: {question_full_text}\n(Error: Prompt template missing)"""
            return default_prompts.DEFAULT_PROMPT_TEMPLATE.format(vignette=vignette, question_full_text=question_full_text)


def run_default_strategy(data: list[dict], model_config_item: dict) -> list[dict]:
    """
    Processes each question entry using the specified LLM, expecting only a single letter answer.
    This is the original processing logic.
    """
    results = []
    total_questions = len(data)
    config_id = model_config_item.get("config_id", model_config_item.get("model_id"))
    model_type = model_config_item.get("type")
    
    
    prompt_type_to_use = model_config_item.get("prompt_strategy_type", "default") 

    loading_animation = None
    
    for thread in threading.enumerate():
        if hasattr(thread, '_target') and thread._target and 'LoadingAnimation' in str(thread._target):
            frame = sys._current_frames().get(thread.ident)
            if frame:
                for local_var in frame.f_locals.values():
                    if isinstance(local_var, ui_utils.LoadingAnimation):
                        loading_animation = local_var
                        break
            if loading_animation: 
                break


    for i, entry in enumerate(data):
        if loading_animation:
            loading_animation.update_progress(i + 1, total_questions)

        logger.info(f"Processing question {i + 1}/{total_questions} with model {config_id} (Default Strategy, Prompt: {prompt_type_to_use})...")
        
        
        prompt = generate_prompt_for_default_strategy(entry, model_type=model_type, prompt_type=prompt_type_to_use)
        
        
        llm_data = llm_clients.get_llm_response(prompt, model_config_item)

        llm_answer = ""
        is_correct = None
        response_time = None
        current_input_tokens = None
        current_output_tokens = None
        answer_length = 0

        if llm_data and llm_data.get('response_json'):
            parsed_response = llm_data['response_json']
            response_time = llm_data.get('response_time')
            current_input_tokens = llm_data.get('input_tokens')
            current_output_tokens = llm_data.get('output_tokens')

            llm_answer = parsed_response.get("answer", "").strip().upper()
            answer_length = len(llm_answer)

            if "error_message" in llm_data: 
                logger.warning(f"Q {i + 1} ({config_id}): Problem with LLM response: {llm_data['error_message']}")

            correct_answer_str = str(entry.get('correctAnswer', '')).strip().upper()
            if not correct_answer_str or "PLACEHOLDER" in correct_answer_str:
                logger.warning(f"Q {i + 1} ({config_id}): Correct answer missing/placeholder. Cannot evaluate correctness.")
                is_correct = None
            elif llm_answer == "X": 
                is_correct = False
                logger.info(f"Q {i + 1} ({config_id}): LLM response parsed as 'X'. Marked as incorrect.")
            elif not llm_answer or llm_answer not in ["A", "B", "C", "D"]: 
                is_correct = False
                logger.warning(f"Q {i + 1} ({config_id}): LLM provided an empty or non-standard invalid answer ('{llm_answer}'). Marked as incorrect.")
            else:
                is_correct = (llm_answer == correct_answer_str)

            logger.info(f"Q {i + 1} ({config_id}): LLM Ans: '{llm_answer}', Correct: '{correct_answer_str}', Match: {is_correct}, Time: {response_time:.2f}s" if response_time is not None else f"Q {i+1} ({config_id}): LLM Ans: '{llm_answer}', Correct: '{correct_answer_str}', Match: {is_correct}")
        else:
            if llm_data: 
                response_time = llm_data.get('response_time', 0)
                logger.error(f"Q {i + 1} ({config_id}): Failed to parse LLM response or extract answer after {response_time:.2f}s.")
            else: 
                logger.error(f"Q {i + 1} ({config_id}): Failed to get any valid LLM response (API call likely failed).")
            llm_answer = "ERROR"
            is_correct = False
            response_time = llm_data.get('response_time') if llm_data else None


        updated_entry = entry.copy()
        updated_entry.update({
            'LLM_answer': llm_answer,
            'is_correct': is_correct,
            'response_time': response_time,
            'input_tokens': current_input_tokens,
            'output_tokens': current_output_tokens,
            'answer_length': answer_length,
            'prompt_strategy': 'default', 
            'prompt_type_used': prompt_type_to_use 
        })
        results.append(updated_entry)
    return results 