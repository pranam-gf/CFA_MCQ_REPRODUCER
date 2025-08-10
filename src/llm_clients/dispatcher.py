"""
Main dispatcher that routes LLM requests to appropriate client modules.
"""
import json
import re
import string
import logging
from .openai_client import get_openai_response
from .anthropic_client import get_anthropic_response
from .xai_client import get_xai_response
from .gemini_client import get_gemini_response
from .groq_client import get_groq_response
from .mistral_client import get_mistral_response
from .writer_client import get_writer_response

logger = logging.getLogger(__name__)


def get_llm_response(prompt: str | list[dict[str, str]], model_config: dict, is_json_response_expected: bool = False) -> dict:
    """
    Route LLM requests to appropriate client based on model type.
    
    Args:
        prompt: The prompt string or list of messages
        model_config: Model configuration dictionary
        is_json_response_expected: Whether to expect JSON response format
        
    Returns:
        Standardized response dictionary with parsed content
    """
    model_type = model_config.get("type")
    config_id = model_config.get("config_id", model_config.get("model_id"))
    
    logger.info(f"Routing request for {model_type} model: {config_id}")
    
    if model_type == "openai":
        raw_response = get_openai_response(prompt, model_config, is_json_response_expected)
    elif model_type == "anthropic":
        raw_response = get_anthropic_response(prompt, model_config, is_json_response_expected)
    elif model_type == "xai":
        raw_response = get_xai_response(prompt, model_config, is_json_response_expected)
    elif model_type == "gemini":
        raw_response = get_gemini_response(prompt, model_config, is_json_response_expected)
    elif model_type == "groq":
        raw_response = get_groq_response(prompt, model_config, is_json_response_expected)
    elif model_type == "mistral_official":
        raw_response = get_mistral_response(prompt, model_config, is_json_response_expected)
    elif model_type == "writer":
        raw_response = get_writer_response(prompt, model_config, is_json_response_expected)
    else:
        logger.error(f"Unsupported model type: {model_type}")
        return {"error_message": f"Unsupported model type: {model_type}", "response_time": 0}
    
    if raw_response.get("error_message"):
        return raw_response
    
    response_text = raw_response.get("response_text", "")
    parsed_content = None
    
    if is_json_response_expected:
        parsed_content = _parse_json_response(response_text, config_id)
        if not parsed_content:
            return {
                "response_json": {"answer": "X", "explanation": "JSON parsing failed"},
                "response_content": "X",
                "error_message": "JSON parsing failed",
                "raw_response_text": response_text,
                **{k: v for k, v in raw_response.items() if k not in ["response_text"]}
            }
    else:
        cleaned_answer = _extract_single_letter_answer(response_text, config_id)
        parsed_content = {"answer": cleaned_answer, "explanation": ""}
    
    return {
        "response_content": parsed_content,
        "response_json": parsed_content if is_json_response_expected else None,
        "raw_response_text": response_text,
        "extracted_letter": parsed_content.get("answer", "X"),
        "parsed_json_response": parsed_content if is_json_response_expected else None,
        "full_response_data": {},
        "latency_ms": raw_response.get("response_time", 0) * 1000,
        "input_tokens": raw_response.get("input_tokens", 0),
        "output_tokens": raw_response.get("output_tokens", 0),
        "reasoning_tokens": raw_response.get("reasoning_tokens", 0),
        "error_message": None,
        "model_config_id": config_id
    }

def _parse_json_response(response_text: str, config_id: str) -> dict | None:
    """Parse JSON response from LLM output."""
    if not response_text:
        return None
        
    try:
        json_match = re.search(r"```json\s*([\\s\\S]*?)\s*```|({.*?})|(\[.*?\])", response_text, re.DOTALL)
        if json_match:
            json_str = next(g for g in json_match.groups() if g is not None)
            return json.loads(json_str)
        else:
            return json.loads(response_text)
    except json.JSONDecodeError:
        logger.error(f"Failed to parse JSON response for {config_id}")
        return None

def _extract_single_letter_answer(response_text: str, config_id: str) -> str:
    """Extract single letter answer (A, B, C, D) from response text."""
    if not response_text:
        logger.error(f"Empty response text for {config_id}")
        return "X"
    
    search_text = response_text[-100:]
    logger.info(f"Extracting answer from last 100 characters for {config_id}")
    
    final_answer_match = re.search(r"Final Answer:\s*([A-D])", search_text, re.IGNORECASE | re.MULTILINE)
    if final_answer_match:
        answer = final_answer_match.group(1).upper()
        logger.info(f"Extracted final answer '{answer}' for {config_id} from explicit 'Final Answer' pattern")
        return answer
    
    escaped_punctuation = re.escape(string.punctuation)
    delimited_match = re.search(
        rf"(?:^|\s|[{escaped_punctuation}])([A-D])(?:$|\s|[{escaped_punctuation}])",
        search_text,
        re.MULTILINE
    )
    if delimited_match:
        answer = delimited_match.group(1).upper()
        logger.info(f"Extracted answer '{answer}' for {config_id} from delimited pattern")
        return answer
    
    fallback_match = re.search(r"([A-D])", search_text)
    if fallback_match:
        answer = fallback_match.group(1).upper()
        logger.warning(f"Used fallback regex to extract answer '{answer}' for {config_id}")
        return answer
    
    logger.error(f"Could not extract answer for {config_id} from: '{search_text}'")
    return "X"