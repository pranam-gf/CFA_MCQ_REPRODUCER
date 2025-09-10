"""
Google Gemini client for handling Gemini models.
"""
import time
import logging
try:
    import google.generativeai as genai
    from google.generativeai import types
except ImportError:
    genai = None
    types = None
from .. import config

logger = logging.getLogger(__name__)


def get_gemini_response(prompt: str | list[dict[str, str]], model_config: dict, is_json_response_expected: bool = False) -> dict:
    """
    Handle Google Gemini API calls.
    
    Args:
        prompt: The prompt string or list of messages
        model_config: Model configuration dictionary
        is_json_response_expected: Whether to expect JSON response format
        
    Returns:
        Dict with response data or error information
    """
    model_id = model_config.get("model_id")
    config_id = model_config.get("config_id", model_id)
    parameters = model_config.get("parameters", {}).copy()
    
    start_time = time.time()
    
    if genai is None:
        logger.error(f"Google Generative AI library not available for model {config_id}")
        return {"error_message": "Google Generative AI library not installed", "response_time": time.time() - start_time}
    
    if not config.GEMINI_API_KEY:
        logger.error(f"Missing Gemini API key for model {config_id}")
        return {"error_message": "Missing Gemini API key", "response_time": 0}
    
    logger.info(f"Calling Gemini API for {config_id}")
    
    gemini_client = genai.Client(api_key=config.GEMINI_API_KEY)
    
    
    gen_config_args = {
        k: v for k, v in parameters.items()
        if k in ["temperature", "top_p", "top_k", "max_output_tokens", "candidate_count", "stop_sequences"] and v is not None
    }
    
    thinking_budget = parameters.get("thinking_budget")
    final_config = None
    
    if thinking_budget is not None:
        try:
            thinking_config = types.ThinkingConfig(thinking_budget=thinking_budget)
            final_config = types.GenerateContentConfig(
                **gen_config_args,
                thinking_config=thinking_config
            )
            logger.info(f"Gemini model {config_id}: Using ThinkingConfig with budget {thinking_budget}")
        except (AttributeError, Exception) as e:
            logger.warning(f"Could not create ThinkingConfig for {config_id}: {e}. Falling back to basic config")
            final_config = gen_config_args
    else:
        final_config = gen_config_args if gen_config_args else None
        logger.info(f"Gemini model {config_id}: Using basic config (no thinking budget)")
    
    if isinstance(prompt, list):
        prompt_text = " ".join([msg.get("content", "") for msg in prompt if isinstance(msg, dict) and msg.get("content")])
    else:
        prompt_text = prompt
    
    try:
        logger.debug(f"Gemini API call for {config_id} with config: {final_config}")
        
        api_response = gemini_client.models.generate_content(
            model=model_id,
            contents=[prompt_text],
            config=final_config
        )
        
        response_text = ""
        if hasattr(api_response, 'text') and api_response.text:
            response_text = api_response.text.strip()
        else:
            logger.warning(f"No text content in Gemini response for {config_id}")
        
        input_tokens = 0
        output_tokens = 0
        
        if hasattr(api_response, 'usage_metadata') and api_response.usage_metadata:
            usage_meta = api_response.usage_metadata
            input_tokens = getattr(usage_meta, 'prompt_token_count', 0) or 0
            
            thoughts_tokens = getattr(usage_meta, 'thoughts_token_count', 0) or 0
            candidates_tokens = getattr(usage_meta, 'candidates_token_count', 0) or 0
            output_tokens = thoughts_tokens + candidates_tokens
            
            if thoughts_tokens > 0:
                logger.info(f"Gemini usage for {config_id}: Input={input_tokens}, Candidates={candidates_tokens}, Thoughts={thoughts_tokens}, Total Output={output_tokens}")
            else:
                logger.info(f"Gemini usage for {config_id}: Input={input_tokens}, Output={output_tokens} (no thoughts)")
        else:
            logger.warning(f"No usage metadata available from Gemini API for {config_id}")
        
        elapsed_time = time.time() - start_time
        logger.info(f"Gemini API call for {config_id} completed in {elapsed_time:.2f}s. Response length: {len(response_text)}")
        
        return {
            "response_text": response_text,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "reasoning_tokens": 0,  
            "response_time": elapsed_time,
            "error_message": None,
            "model_config_id": config_id
        }
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        logger.error(f"Gemini API Error for {config_id} after {elapsed_time:.2f}s: {e}", exc_info=True)
        return {
            "error_message": str(e),
            "response_time": elapsed_time,
            "response_text": "",
            "details": {"type": type(e).__name__, "unexpected": True}
        }