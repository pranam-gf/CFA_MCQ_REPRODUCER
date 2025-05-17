"""
Self-consistency prompting strategy.
"""
import logging
import random
import re
import sys
import threading
from collections import Counter

from .. import llm_clients 
from ..utils import ui_utils
from ..utils.prompt_utils import parse_question_data

logger = logging.getLogger(__name__)

def extract_choice_letter_from_cot(text: str, model_config_id: str = "unknown_model_cot") -> str | None:
    """Extracts the final choice letter (A, B, C, D) from a CoT response, focusing on the last 100 characters."""
    if not text:
        return None

    search_text = text[-100:]
    logger.debug(f"CoT-specific extraction for {model_config_id} focusing on last 100 chars: '{search_text}...'")
    header_pattern = r"^(?:\\#\\#\\# Conclusion|Conclusion:|\\*\\*Conclude:|Conclude:|Final Answer:|The final answer is:|My final choice is|The best choice is|The correct option is)[\\s\\S]*?"
    answer_formats_after_header = [
        r"\\*\\*([A-D])\\*\\*[:\\.\\)]?",        
        r"\\s*([A-D])\\b",                      
        r"\\b([A-D])[:\\.\\)]",               
        r"letter\\s+([A-D])\\b",             
        r"is\\s+([A-D])\\b"                  
    ]
    for ans_fmt in answer_formats_after_header:
        full_pattern = header_pattern + ans_fmt
        conclusion_match = re.search(full_pattern, search_text, re.IGNORECASE | re.MULTILINE)
        if conclusion_match:
            letter = conclusion_match.group(1).upper()
            logger.debug(f"Extracted letter '{letter}' using Pattern 1 ({ans_fmt=}) from CoT (last 100 chars) for {model_config_id}")
            return letter

    direct_choice_patterns = [
        r"^\s*(?:-|\*\*|[•●])?\s*\*\*([A-D])\*\*[:\.\)]?", 
        r"^\s*(?:-|\*\*|[•●])?\s*([A-D])[:\.\)]\s*(?:is the (?:best|correct)|The (?:best|correct))", 
        r"^\s*(?:-|\*\*|[•●])?\s*\*\*?([A-D])\*\*?[:\.\)]", 
        r"^\s*The (?:best|correct|final) (?:answer|option|choice) is\s*\*\*([A-D])\*\*", 
        r"^\s*The (?:best|correct|final) (?:answer|option|choice) is\s+([A-D])\b"    
    ]
    for pattern in direct_choice_patterns:
        match = re.search(pattern, search_text, re.IGNORECASE | re.MULTILINE)
        if match:
            
            for group_val in match.groups(): 
                if group_val:
                    letter = group_val.upper()
                    logger.debug(f"Extracted letter '{letter}' using Pattern 2 ({pattern=}) from CoT (last 100 chars) for {model_config_id}")
                    return letter
    general_keyword_match = re.search(
        r"(?:Answer|Choice|Option) DNE\b" + 
        r"|(?:Final Answer|The answer is|Best choice is|The final choice is|Conclude by selecting|My choice is|The correct option is|Option is|Answer)[:\s]*[\s\S]*?\*\*([A-D])\*\*" +
        r"|(?:Final Answer|The answer is|Best choice is|The final choice is|Conclude by selecting|My choice is|The correct option is|Option is|Answer)[:\s]*[\s\S]*?\b([A-D])\b" +
        r"|\b([A-D])\.\s*\(?\w*\s*\w*\s*\w*\)?(?:\s+is correct|\s+is the answer)?" + 
        r"|\b([A-D])\s+is the correct answer\b" +
        r"|\b(?:The best answer is letter |The final answer is |My final choice is )([A-D])\b",
        search_text,
        re.IGNORECASE | re.MULTILINE
    )
    if general_keyword_match:
        for group in general_keyword_match.groups():
            if group:
                letter = group.upper()
                logger.debug(f"Extracted letter '{letter}' using Pattern 3 (general keywords) from CoT (last 100 chars) for {model_config_id}")
                return letter
    
    end_of_text_match = re.search(
        r"(?:\b([A-D])\s*[\.\!\?]?\s*$)" + 
        r"|(?:^\s*([A-D])\s*[\.\!\?]?\s*$)",  
        search_text.strip(), 
        re.IGNORECASE | re.MULTILINE
    )
    if end_of_text_match:
        for group in end_of_text_match.groups():
            if group:
                letter = group.upper()
                logger.debug(f"Extracted letter '{letter}' using Pattern 4 (end_of_text_match) from CoT (last 100 chars) for {model_config_id}")
                return letter

    logger.warning(f"Could not extract choice letter from CoT response (last 100 chars) for {model_config_id}: '{search_text}...'")
    return None 

def generate_prompt_for_cot_strategy(entry: dict, cot_template: str) -> str:
    """Generates a CoT prompt for a given question entry using the standardized parser."""
    parsed_data = parse_question_data(entry)
    
    return cot_template.format(
        vignette=parsed_data['vignette'],
        question_stem=parsed_data['question_stem'],
        option_a=parsed_data['options_dict'].get('A', 'Option A not provided'),
        option_b=parsed_data['options_dict'].get('B', 'Option B not provided'),
        option_c=parsed_data['options_dict'].get('C', 'Option C not provided')
    )

def run_self_consistency_strategy(data: list[dict], model_config_item: dict, cot_template: str, n_samples: int = 3) -> list[dict]:
    """
    Processes questions using CoT prompts and self-consistency (majority vote).
    """
    results_for_all_questions = []
    total_questions = len(data)
    config_id = model_config_item.get("config_id", model_config_item.get("model_id"))
    model_type = model_config_item.get("type") 
    sampling_params = model_config_item.get("parameters", {}).copy()
    sampling_params['temperature'] = sampling_params.get('temperature_cot_sampling', 0.7) 
    
    if 'max_tokens_cot' in sampling_params:
        sampling_params['max_tokens'] = sampling_params['max_tokens_cot']
    elif 'max_tokens' not in sampling_params:
        sampling_params['max_tokens'] = 4000
    
    sampling_params.pop('response_format', None)

    loading_animation = None
    for thread in threading.enumerate():
        if hasattr(thread, '_target') and thread._target and 'LoadingAnimation' in str(thread._target):
            frame = sys._current_frames().get(thread.ident)
            if frame:
                loading_animation = next((lv for lv in frame.f_locals.values() if isinstance(lv, ui_utils.LoadingAnimation)), None)
            if loading_animation: break

    for i, entry in enumerate(data):
        if loading_animation:
            loading_animation.update_progress(i + 1, total_questions)
            loading_animation.message = f"Processing Q{i+1}/{total_questions} with {config_id} (CoT Samples)"

        logger.info(f"Processing Q {i + 1}/{total_questions} with {config_id} (Self-Consistency, {n_samples} samples)...")
        prompt_for_entry = generate_prompt_for_cot_strategy(entry, cot_template)

        sample_responses = [] 
        all_llm_answers_this_question = []

        for s_idx in range(n_samples):
            if loading_animation:
                loading_animation.message = f"Processing Q{i+1}/{total_questions} with {config_id} (Sample {s_idx+1}/{n_samples})"
            
            logger.info(f"  Sample {s_idx + 1}/{n_samples} for Q {i+1}...")
            
            current_call_model_config = model_config_item.copy()
            current_call_model_config["parameters"] = sampling_params
            
            
            llm_call_data = llm_clients.get_llm_response(prompt_for_entry, current_call_model_config, is_json_response_expected=False)

            if llm_call_data and not llm_call_data.get("error_message"):
                raw_text = llm_call_data.get('raw_response_text', '')
                letter = extract_choice_letter_from_cot(raw_text, config_id)
                if letter: 
                    all_llm_answers_this_question.append(letter)
                
                sample_responses.append({
                    "extracted_letter": letter or "PARSE_FAIL",
                    "raw_text": raw_text,
                    "response_time": llm_call_data.get('response_time'),
                    "input_tokens": llm_call_data.get('input_tokens'),
                    "output_tokens": llm_call_data.get('output_tokens'),
                })
            else:
                logger.error(f"    Failed to get valid response for sample {s_idx + 1} of Q {i+1}: {llm_call_data.get('error_message', 'Unknown error') if llm_call_data else 'No response'}")
                sample_responses.append({
                    "extracted_letter": "API_FAIL",
                    "raw_text": llm_call_data.get('raw_response_text', 'ERROR_NO_RAW_TEXT') if llm_call_data else "ERROR_NO_RESPONSE",
                    "response_time": llm_call_data.get('response_time'),
                    "input_tokens": llm_call_data.get('input_tokens'),
                    "output_tokens": llm_call_data.get('output_tokens'),
                    "error": llm_call_data.get('error_message', 'Unknown error') if llm_call_data else 'No response'
                })
        
        final_voted_answer = "ERROR_NO_VOTES"
        vote_tally_dict = {}
        if all_llm_answers_this_question:
            tally = Counter(all_llm_answers_this_question)
            vote_tally_dict = dict(tally)
            if tally:
                final_voted_answer = tally.most_common(1)[0][0]
                logger.info(f"  Q {i+1} ({config_id}) - Vote Tally: {vote_tally_dict}, Final Answer: {final_voted_answer}")
            else:
                logger.warning(f"  Q {i+1} ({config_id}) - No valid answers to tally. All samples failed parsing.")
                final_voted_answer = "ERROR_ALL_PARSE_FAIL"
        else:
            logger.error(f"  Q {i+1} ({config_id}) - All {n_samples} samples failed to produce a response or parseable letter.")
            final_voted_answer = "ERROR_ALL_API_FAIL"

        correct_answer_str = str(entry.get('correctAnswer', '')).strip().upper()
        is_correct = None
        if final_voted_answer.startswith("ERROR"):
            is_correct = False
        elif not correct_answer_str or "PLACEHOLDER" in correct_answer_str:
            logger.warning(f"Q {i + 1} ({config_id}): Correct answer missing/placeholder. Cannot evaluate correctness for CoT.")
        else:
            is_correct = (final_voted_answer == correct_answer_str)

        avg_response_time = sum(s.get('response_time', 0) for s in sample_responses if s.get('response_time')) / len(sample_responses) if sample_responses else 0
        total_input_tokens = sum(s.get('input_tokens', 0) for s in sample_responses if s.get('input_tokens') is not None)
        total_output_tokens = sum(s.get('output_tokens', 0) for s in sample_responses if s.get('output_tokens') is not None)

        logger.info(f"Q {i+1} ({config_id}) CoT: Final Voted Ans: '{final_voted_answer}', Correct: '{correct_answer_str}', Match: {is_correct}, Avg Time/Sample: {avg_response_time:.2f}s")

        updated_entry = entry.copy()
        updated_entry.update({
            'LLM_answer': final_voted_answer,
            'is_correct': is_correct,
            'response_time': avg_response_time * n_samples, 
            'avg_response_time_per_sample': avg_response_time,
            'input_tokens': total_input_tokens,
            'output_tokens': total_output_tokens,
            'answer_length': len(final_voted_answer),
            'prompt_strategy': f'self_consistency_cot_n{n_samples}',
            'vote_tally': vote_tally_dict,
            'all_samples_details': sample_responses 
        })
        results_for_all_questions.append(updated_entry)

    if loading_animation: 
        loading_animation.message = f"Processing with {config_id}"

    return results_for_all_questions 