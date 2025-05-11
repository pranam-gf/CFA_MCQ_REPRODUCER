"""
Self-consistency prompting strategy.
"""
import logging
import random
import re
import sys
import threading
from collections import Counter

from .. import llm_clients # Adjusted import
from ..prompts import cot as cot_prompts # Adjusted import
from .. import ui_utils # Adjusted import

logger = logging.getLogger(__name__)

def extract_choice_letter_from_cot(text: str) -> str | None:
    """Extracts the final choice letter (A, B, C, D) from a CoT response."""

    # Pattern 1: More flexible header matching, then looking for the letter.
    # Handles "### Conclusion", "Conclusion:", "**Conclude:**", "Final Answer:", etc.
    # and then looks for various ways the answer might be presented.
    header_pattern = r"^(?:\#\#\# Conclusion|Conclusion:|\*\*Conclude:|Conclude:|Final Answer:|The final answer is:|My final choice is|The best choice is|The correct option is)[\s\S]*?"
    answer_formats_after_header = [
        r"\*\*([A-D])\*\*[:\.\)]?",            # e.g., **A**: or **A**. or **A**)
        r"\b([A-D])[:\.\)]",                 # e.g., A: or A. or A)
        r"letter\s+([A-D])\b",             # e.g., letter A
        r"is\s+([A-D])\b"                  # e.g., is A
    ]
    for ans_fmt in answer_formats_after_header:
        full_pattern = header_pattern + ans_fmt
        conclusion_match = re.search(full_pattern, text, re.IGNORECASE | re.MULTILINE)
        if conclusion_match:
            letter = conclusion_match.group(1).upper()
            logger.debug(f"Extracted letter '{letter}' using Pattern 1 ({ans_fmt=}) from CoT: ...{text[-180:]}")
            return letter

    # Pattern 2: Looks for lines that start with a clear indication of the answer choice itself.
    # e.g., "- **A:** ...", "**B.** ...", "C) ... is the best answer"
    direct_choice_patterns = [
        r"^\s*(?:-|\*\*|[•●])?\s*\*\*([A-D])\*\*[:\.\)]?", # e.g., - **A:** or **B.**
        r"^\s*(?:-|\*\*|[•●])?\s*([A-D])[:\.\)]\s*(?:is the (?:best|correct)|The (?:best|correct))", # e.g., A) is the best or - B. The correct
        r"^\s*(?:-|\*\*|[•●])?\s*\*\*?([A-D])\*\*?[:\.\)]", # Added: More general, captures bold or non-bold letter with punctuation after list marker
        r"^\s*The (?:best|correct|final) (?:answer|option|choice) is\s*\*\*([A-D])\*\*", # e.g. The best answer is **A**
        r"^\s*The (?:best|correct|final) (?:answer|option|choice) is\s+([A-D])\b"    # e.g. The correct choice is A
    ]
    for pattern in direct_choice_patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            # Find the first non-None group, as some patterns might have multiple optional groups before the letter
            for group_val in match.groups(): 
                if group_val:
                    letter = group_val.upper()
                    logger.debug(f"Extracted letter '{letter}' using Pattern 2 ({pattern=}) from CoT: ...{text[-180:]}")
                    return letter

    # Pattern 3: General keywords (adapted from original general_match)
    # This is less specific about line start or explicit headers.
    general_keyword_match = re.search(
        r"(?:Answer|Choice|Option) DNE\b" + # Handle "Does Not Exist" as a special case if needed, though not A-D
        # Allow more characters between keyword and letter for bolded version
        r"|(?:Final Answer|The answer is|Best choice is|The final choice is|Conclude by selecting|My choice is|The correct option is|Option is|Answer)[:\s]*[\s\S]*?\*\*([A-D])\*\*" +
        # Allow more characters between keyword and letter for non-bolded version
        r"|(?:Final Answer|The answer is|Best choice is|The final choice is|Conclude by selecting|My choice is|The correct option is|Option is|Answer)[:\s]*[\s\S]*?\b([A-D])\b" +
        r"|\b([A-D])\.\s*\(?\w*\s*\w*\s*\w*\)?(?:\s+is correct|\s+is the answer)?" + 
        r"|\b([A-D])\s+is the correct answer\b" +
        r"|\b(?:The best answer is letter |The final answer is |My final choice is )([A-D])\b",
        text,
        re.IGNORECASE | re.MULTILINE
    )
    if general_keyword_match:
        for group in general_keyword_match.groups():
            if group:
                letter = group.upper()
                logger.debug(f"Extracted letter '{letter}' using Pattern 3 (general keywords) from CoT: ...{text[-100:]}")
                return letter
    
    # Pattern 4: Fallback: a letter A, B, C, or D at the very end of the text or a line.
    end_of_text_match = re.search(
        r"(?:\b([A-D])\s*[\.\!\?]?\s*$)" + 
        r"|(?:^\s*([A-D])\s*[\.\!\?]?\s*$)",  
        text.strip(), 
        re.IGNORECASE | re.MULTILINE
    )
    if end_of_text_match:
        for group in end_of_text_match.groups():
            if group:
                letter = group.upper()
                logger.debug(f"Extracted letter '{letter}' using Pattern 4 (end_of_text_match) from CoT: ...{text[-100:]}")
                return letter

    logger.warning(f"Could not extract choice letter from CoT response: ...{text[-180:]}")
    return None # Return None if no letter is found

def generate_prompt_for_cot_strategy(entry: dict) -> str:
    """Generates a CoT prompt for a given question entry using pre-structured option fields."""
    
    vignette_text = entry.get('vignette', 'Vignette not available.')
    question_stem = entry.get('question', 'Question stem not available.')
    
    opt_a = entry.get('option_a', 'N/A')
    opt_b = entry.get('option_b', 'N/A')
    opt_c = entry.get('option_c', 'N/A')
    opt_d = entry.get('option_d') # Allow opt_d to be None if not present

    # The COHERENT_CFA_COT prompt in src/prompts/cot.py expects:
    # {vignette}, {question_stem}, {opt_a}, {opt_b}, {opt_c}, {opt_d}
    
    return cot_prompts.COHERENT_CFA_COT.format(
        vignette=vignette_text,
        question_stem=question_stem,
        opt_a=opt_a,
        opt_b=opt_b,
        opt_c=opt_c,
        opt_d=opt_d if opt_d is not None else "N/A (Not applicable)" # Ensure opt_d is a string
    )

def run_self_consistency_strategy(data: list[dict], model_config_item: dict, n_samples: int = 3) -> list[dict]:
    """
    Processes questions using CoT prompts and self-consistency (majority vote).
    """
    results_for_all_questions = []
    total_questions = len(data)
    config_id = model_config_item.get("config_id", model_config_item.get("model_id"))
    model_type = model_config_item.get("type") # For prompt generation if needed

    # Adjust model parameters for sampling (e.g., temperature)
    sampling_params = model_config_item.get("parameters", {}).copy()
    sampling_params['temperature'] = sampling_params.get('temperature_cot_sampling', 0.7) # Use a specific temp or default
    # Max tokens might need to be higher for CoT responses
    sampling_params['max_tokens'] = sampling_params.get('max_tokens_cot', 4000) # Increased from 500 to 1024
    # Ensure response_format is NOT json_object for CoT text generation
    sampling_params.pop('response_format', None)

    # Find loading animation instance
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
        
        # This prompt generation might need vignette + question stem + options formatted correctly
        # for the COHERENT_CFA_COT template.
        # The `entry` dict structure is key here.
        prompt_template_for_entry = generate_prompt_for_cot_strategy(entry)

        sample_responses = [] # To store (letter, raw_text, response_time, in_tokens, out_tokens)
        all_llm_answers_this_question = []

        for s_idx in range(n_samples):
            if loading_animation:
                loading_animation.message = f"Processing Q{i+1}/{total_questions} with {config_id} (Sample {s_idx+1}/{n_samples})"
            
            logger.info(f"  Sample {s_idx + 1}/{n_samples} for Q {i+1}...")
            
            # Create a unique model_config for this sample call if params differ significantly, 
            # or just pass the sampling_params to get_llm_response if it accepts them as overrides.
            # For now, assuming get_llm_response uses params from model_config_item and we can update it.
            current_call_model_config = model_config_item.copy()
            current_call_model_config["parameters"] = sampling_params
            
            # get_llm_response expects is_json_response_expected = False for CoT text
            llm_call_data = llm_clients.get_llm_response(prompt_template_for_entry, current_call_model_config, is_json_response_expected=False)

            if llm_call_data and not llm_call_data.get("error_message"):
                raw_text = llm_call_data.get('raw_response_text', '')
                letter = extract_choice_letter_from_cot(raw_text)
                if letter: # Only count valid extractions
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
            'response_time': avg_response_time * n_samples, # Or total time across samples
            'avg_response_time_per_sample': avg_response_time,
            'input_tokens': total_input_tokens,
            'output_tokens': total_output_tokens,
            'answer_length': len(final_voted_answer), # Length of the letter, or error string
            'prompt_strategy': f'self_consistency_cot_n{n_samples}',
            'vote_tally': vote_tally_dict,
            'all_samples_details': sample_responses # Storing raw texts and individual parse results
        })
        results_for_all_questions.append(updated_entry)

    if loading_animation: # Reset message after this model is done
        loading_animation.message = f"Processing with {config_id}"

    return results_for_all_questions 